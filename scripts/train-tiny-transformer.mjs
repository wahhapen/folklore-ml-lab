import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import * as tf from "@tensorflow/tfjs";

const root = process.cwd();
const dataRoot = path.join(root, "ml/data/edition-fingerprint-v1");
const runRoot = path.join(root, "ml/runs/tiny-byte-transformer-v1");
const releasesRoot = path.join(root, "data/derived/releases");
const releaseCandidates = (
  await readdir(releasesRoot, { withFileTypes: true })
)
  .filter((entry) => entry.isDirectory() && entry.name.startsWith("corpus-v"))
  .map((entry) => path.join(releasesRoot, entry.name));
const releaseRoot = process.env.FOLKLORE_CORPUS_DIR
  ? path.resolve(process.env.FOLKLORE_CORPUS_DIR)
  : releaseCandidates.length === 1
    ? releaseCandidates[0]
    : (() => {
        throw new Error(
          `Expected exactly one installed Corpus Release, found ${releaseCandidates.length}. Set FOLKLORE_CORPUS_DIR explicitly.`,
        );
      })();
const seed = 20260724;
const config = {
  vocabSize: 257,
  eosToken: 256,
  context: 64,
  width: 32,
  heads: 4,
  feedForward: 64,
  batchSize: 8,
  steps: Number(process.env.FOLKLORE_TINY_STEPS ?? 600),
  validationEvery: 50,
  learningRate: 0.002,
  gradientClip: 1,
};

function parseJsonLines(contents) {
  return contents
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function mulberry32(initialSeed) {
  let state = initialSeed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function streamFromRows(rows) {
  const values = [];
  for (const row of rows) {
    values.push(...Buffer.from(row.text, "utf8"), config.eosToken);
  }
  return Int32Array.from(values);
}

function makeBatch(stream, random, batchSize = config.batchSize) {
  const inputs = new Int32Array(batchSize * config.context);
  const targets = new Int32Array(batchSize * config.context);
  const maximumStart = stream.length - config.context - 1;
  for (let batch = 0; batch < batchSize; batch += 1) {
    const start = Math.floor(random() * maximumStart);
    for (let index = 0; index < config.context; index += 1) {
      inputs[batch * config.context + index] = stream[start + index];
      targets[batch * config.context + index] = stream[start + index + 1];
    }
  }
  return {
    inputs: tf.tensor2d(inputs, [batchSize, config.context], "int32"),
    targets: tf.tensor2d(targets, [batchSize, config.context], "int32"),
  };
}

function layerNorm(value, gamma, beta) {
  const { mean, variance } = tf.moments(value, -1, true);
  return value
    .sub(mean)
    .mul(tf.rsqrt(variance.add(1e-5)))
    .mul(gamma)
    .add(beta);
}

function gelu(value) {
  return value
    .mul(0.5)
    .mul(tf.erf(value.div(Math.sqrt(2))).add(1));
}

function causalMask(length) {
  const values = new Float32Array(length * length);
  for (let row = 0; row < length; row += 1) {
    for (let column = row + 1; column < length; column += 1) {
      values[row * length + column] = -1e9;
    }
  }
  return tf.tensor4d(values, [1, 1, length, length]);
}

function linear3d(value, weight, bias) {
  const [batchSize, length] = value.shape;
  let output = value
    .reshape([batchSize * length, value.shape[2]])
    .matMul(weight);
  if (bias) output = output.add(bias);
  return output.reshape([batchSize, length, weight.shape[1]]);
}

function createModel() {
  let initializerSeed = seed;
  const normal = (name, shape) =>
    tf.variable(
      tf.randomNormal(shape, 0, 0.02, "float32", initializerSeed++),
      true,
      name,
    );
  const zeros = (name, shape) =>
    tf.variable(tf.zeros(shape), true, name);
  const ones = (name, shape) => tf.variable(tf.ones(shape), true, name);
  return {
    tokenEmbedding: normal("token_embedding", [config.vocabSize, config.width]),
    positionEmbedding: normal("position_embedding", [config.context, config.width]),
    ln1Gamma: ones("ln1_gamma", [config.width]),
    ln1Beta: zeros("ln1_beta", [config.width]),
    query: normal("attention_query", [config.width, config.width]),
    key: normal("attention_key", [config.width, config.width]),
    value: normal("attention_value", [config.width, config.width]),
    projection: normal("attention_projection", [config.width, config.width]),
    ln2Gamma: ones("ln2_gamma", [config.width]),
    ln2Beta: zeros("ln2_beta", [config.width]),
    feedForwardIn: normal("feed_forward_in", [
      config.width,
      config.feedForward,
    ]),
    feedForwardBias: zeros("feed_forward_bias", [config.feedForward]),
    feedForwardOut: normal("feed_forward_out", [
      config.feedForward,
      config.width,
    ]),
    feedForwardOutBias: zeros("feed_forward_out_bias", [config.width]),
    finalGamma: ones("final_gamma", [config.width]),
    finalBeta: zeros("final_beta", [config.width]),
    outputBias: zeros("output_bias", [config.vocabSize]),
  };
}

function forward(model, inputs) {
  const [batchSize, length] = inputs.shape;
  const headWidth = config.width / config.heads;
  const positions = model.positionEmbedding
    .slice([0, 0], [length, config.width])
    .expandDims(0);
  let state = tf.gather(model.tokenEmbedding, inputs).add(positions);
  const normalized = layerNorm(state, model.ln1Gamma, model.ln1Beta);
  const query = linear3d(normalized, model.query)
    .reshape([batchSize, length, config.heads, headWidth])
    .transpose([0, 2, 1, 3]);
  const key = linear3d(normalized, model.key)
    .reshape([batchSize, length, config.heads, headWidth])
    .transpose([0, 2, 1, 3]);
  const value = linear3d(normalized, model.value)
    .reshape([batchSize, length, config.heads, headWidth])
    .transpose([0, 2, 1, 3]);
  const attentionHeads = query
    .matMul(key, false, true)
    .div(Math.sqrt(headWidth))
    .add(causalMask(length))
    .softmax(-1)
    .matMul(value)
    .transpose([0, 2, 1, 3])
    .reshape([batchSize, length, config.width]);
  const attention = linear3d(attentionHeads, model.projection);
  state = state.add(attention);
  const feedForwardInput = layerNorm(
    state,
    model.ln2Gamma,
    model.ln2Beta,
  );
  const feedForward = linear3d(
    gelu(
      linear3d(
        feedForwardInput,
        model.feedForwardIn,
        model.feedForwardBias,
      ),
    ),
    model.feedForwardOut,
    model.feedForwardOutBias,
  );
  state = state.add(feedForward);
  const finalState = layerNorm(state, model.finalGamma, model.finalBeta);
  return finalState
    .reshape([batchSize * length, config.width])
    .matMul(model.tokenEmbedding, false, true)
    .add(model.outputBias)
    .reshape([batchSize, length, config.vocabSize]);
}

function lossForBatch(model, inputs, targets) {
  const logits = forward(model, inputs).reshape([
    -1,
    config.vocabSize,
  ]);
  const labels = tf.oneHot(targets.reshape([-1]), config.vocabSize);
  return tf.losses.softmaxCrossEntropy(labels, logits).mean();
}

function clippedGradients(gradients) {
  const names = Object.keys(gradients);
  const norm = tf.tidy(() =>
    Math.sqrt(
      tf
        .addN(names.map((name) => gradients[name].square().sum()))
        .dataSync()[0],
    ),
  );
  const scale = Math.min(1, config.gradientClip / Math.max(norm, 1e-12));
  if (scale === 1) return { values: gradients, created: [] };
  const values = Object.fromEntries(
    names.map((name) => [name, gradients[name].mul(scale)]),
  );
  return { values, created: Object.values(values) };
}

function evaluate(model, stream, batchSeed, batches = 8) {
  const random = mulberry32(batchSeed);
  const losses = [];
  for (let index = 0; index < batches; index += 1) {
    const batch = makeBatch(stream, random);
    const loss = tf.tidy(() =>
      lossForBatch(model, batch.inputs, batch.targets).dataSync()[0],
    );
    batch.inputs.dispose();
    batch.targets.dispose();
    losses.push(loss);
  }
  return losses.reduce((sum, loss) => sum + loss, 0) / losses.length;
}

function ngramBaselines(trainStream, validationStream) {
  const vocabulary = config.vocabSize;
  const unigram = new Float64Array(vocabulary).fill(1);
  const bigram = Array.from(
    { length: vocabulary },
    () => new Float64Array(vocabulary).fill(0.1),
  );
  const contextCounts = new Float64Array(vocabulary).fill(vocabulary * 0.1);
  for (let index = 0; index < trainStream.length; index += 1) {
    unigram[trainStream[index]] += 1;
    if (index) {
      bigram[trainStream[index - 1]][trainStream[index]] += 1;
      contextCounts[trainStream[index - 1]] += 1;
    }
  }
  const unigramTotal = unigram.reduce((sum, count) => sum + count, 0);
  let unigramLoss = 0;
  let bigramLoss = 0;
  for (let index = 1; index < validationStream.length; index += 1) {
    const token = validationStream[index];
    unigramLoss -= Math.log(unigram[token] / unigramTotal);
    bigramLoss -= Math.log(
      bigram[validationStream[index - 1]][token] /
        contextCounts[validationStream[index - 1]],
    );
  }
  const denominator = validationStream.length - 1;
  return {
    unigramNatsPerByte: unigramLoss / denominator,
    bigramNatsPerByte: bigramLoss / denominator,
  };
}

function sampleIndex(probabilities, random) {
  const target = random();
  let cumulative = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index];
    if (target <= cumulative) return index;
  }
  return probabilities.length - 1;
}

function generate(model, prompt, temperature, random, length = 64) {
  const tokens = [...Buffer.from(prompt, "utf8")];
  for (let step = 0; step < length; step += 1) {
    const context = tokens.slice(-config.context);
    const probabilities = tf.tidy(() => {
      const inputs = tf.tensor2d(
        Int32Array.from(context),
        [1, context.length],
        "int32",
      );
      const logits = forward(model, inputs)
        .slice([0, context.length - 1, 0], [1, 1, config.vocabSize])
        .reshape([config.vocabSize])
        .div(temperature);
      return logits.softmax().dataSync();
    });
    const token = sampleIndex(probabilities, random);
    if (token === config.eosToken) break;
    tokens.push(token);
  }
  return Buffer.from(tokens.filter((token) => token < 256)).toString("utf8");
}

function longestTrainingSpan(sample, trainingText) {
  for (let length = Math.min(sample.length, 96); length >= 1; length -= 1) {
    for (let start = 0; start <= sample.length - length; start += 1) {
      if (trainingText.includes(sample.slice(start, start + length))) return length;
    }
  }
  return 0;
}

function shingleSet(value, width = 5) {
  return new Set(
    Array.from(
      { length: Math.max(0, value.length - width + 1) },
      (_, index) => value.slice(index, index + width),
    ),
  );
}

function jaccard(left, right) {
  const intersection = [...left].filter((value) => right.has(value)).length;
  return intersection / Math.max(new Set([...left, ...right]).size, 1);
}

await tf.setBackend("cpu");
await tf.ready();
const releaseManifestContents = await readFile(
  path.join(releaseRoot, "manifest.json"),
  "utf8",
);
const releaseManifest = JSON.parse(releaseManifestContents);
const taskManifest = JSON.parse(
  await readFile(path.join(dataRoot, "manifest.json"), "utf8"),
);
const releaseManifestSha256 = createHash("sha256")
  .update(releaseManifestContents)
  .digest("hex");
if (
  taskManifest.corpusRelease !== releaseManifest.releaseId
  || taskManifest.corpusManifestSha256 !== releaseManifestSha256
) {
  throw new Error(
    "Prepared task data is not pinned to the selected Corpus Release.",
  );
}
const trainRows = parseJsonLines(
  await readFile(path.join(dataRoot, "train.jsonl"), "utf8"),
);
const validationRows = parseJsonLines(
  await readFile(path.join(dataRoot, "validation.jsonl"), "utf8"),
);
const trainStream = streamFromRows(trainRows);
const validationStream = streamFromRows(validationRows);
const baselines = ngramBaselines(trainStream, validationStream);
const model = createModel();
const variables = Object.values(model);
const parameterCount = variables.reduce(
  (total, variable) => total + variable.size,
  0,
);
const optimizer = tf.train.adam(config.learningRate);
const trainingRandom = mulberry32(seed);
const history = [];
const startedAt = Date.now();

for (let step = 1; step <= config.steps; step += 1) {
  const batch = makeBatch(trainStream, trainingRandom);
  const { value, grads } = tf.variableGrads(
    () => lossForBatch(model, batch.inputs, batch.targets),
    variables,
  );
  const { values: gradients, created } = clippedGradients(grads);
  optimizer.applyGradients(gradients);
  const trainLoss = value.dataSync()[0];
  value.dispose();
  Object.values(grads).forEach((gradient) => gradient.dispose());
  created.forEach((gradient) => gradient.dispose());
  batch.inputs.dispose();
  batch.targets.dispose();

  if (step === 1 || step % config.validationEvery === 0) {
    const validationLoss = evaluate(model, validationStream, seed + step);
    const record = {
      step,
      trainNatsPerByte: trainLoss,
      validationNatsPerByte: validationLoss,
      validationBitsPerByte: validationLoss / Math.log(2),
      elapsedSeconds: (Date.now() - startedAt) / 1000,
    };
    history.push(record);
    console.log(JSON.stringify(record));
  }
}

const finalValidationLoss = evaluate(model, validationStream, seed + 9999, 16);
const samplePrompts = [
  "Once upon a time",
  "In the forest",
  "The old woman said",
  "A king had three",
  "When the moon rose",
  "The fox answered",
];
const sampleRandom = mulberry32(seed + 42);
const samples = samplePrompts.flatMap((prompt) =>
  [0.7, 1.0].map((temperature) => ({
    prompt,
    temperature,
    text: generate(model, prompt, temperature, sampleRandom),
  })),
);
const trainingText = trainRows.map((row) => row.text).join("\n");
const trainingShingles = shingleSet(trainingText);
const memorization = samples.map((sample) => ({
  prompt: sample.prompt,
  temperature: sample.temperature,
  longestExactTrainingSpan: longestTrainingSpan(sample.text, trainingText),
  containsCopied32ByteSpan: longestTrainingSpan(sample.text, trainingText) >= 32,
  fiveCharacterShingleJaccard: jaccard(
    shingleSet(sample.text),
    trainingShingles,
  ),
}));
const metrics = {
  experiment: "tiny-byte-transformer-v1",
  parameterCount,
  steps: config.steps,
  trainBytes: trainStream.length,
  validationBytes: validationStream.length,
  final: {
    natsPerByte: finalValidationLoss,
    bitsPerByte: finalValidationLoss / Math.log(2),
    bytePerplexity: Math.exp(finalValidationLoss),
  },
  baselines: {
    unigram: {
      natsPerByte: baselines.unigramNatsPerByte,
      bitsPerByte: baselines.unigramNatsPerByte / Math.log(2),
    },
    bigram: {
      natsPerByte: baselines.bigramNatsPerByte,
      bitsPerByte: baselines.bigramNatsPerByte / Math.log(2),
    },
  },
  improvementOverUnigram:
    (baselines.unigramNatsPerByte - finalValidationLoss) /
    baselines.unigramNatsPerByte,
  memorization: {
    longestExactTrainingSpan: Math.max(
      ...memorization.map((row) => row.longestExactTrainingSpan),
    ),
    copied32ByteSampleRate:
      memorization.filter((row) => row.containsCopied32ByteSpan).length /
      memorization.length,
    perSample: memorization,
  },
};
const run = {
  schemaVersion: "folklore-ml-run-v1",
  experiment: "tiny-byte-transformer-v1",
  purpose:
    "Educational byte-level language-model training; not a practically useful folklore model.",
  corpusRelease: releaseManifest.releaseId,
  corpusManifestSha256: releaseManifestSha256,
  datasetSha256: taskManifest.datasetSha256,
  seed,
  backend: await tf.getBackend(),
  tensorflowJsCore: tf.version_core,
  config,
  command: "npm run ml:tiny",
  durationSeconds: (Date.now() - startedAt) / 1000,
};
const checkpoint = {
  format: "tfjs-variable-arrays-v1",
  config,
  weights: Object.fromEntries(
    Object.entries(model).map(([name, variable]) => [
      name,
      { shape: variable.shape, values: Array.from(variable.dataSync()) },
    ]),
  ),
};
const modelCard = `# Tiny Byte Transformer v1

This ${parameterCount.toLocaleString()}-parameter, one-block causal Transformer
was trained from scratch on the training split of Folklore Corpus v0.1.0. It is
an instrumented learning experiment, not a useful language model and not a
candidate for deployment.

- Corpus: \`${releaseManifest.releaseId}\`
- Seed: \`${seed}\`
- Context: ${config.context} bytes
- Width / heads / feed-forward: ${config.width} / ${config.heads} / ${config.feedForward}
- Steps: ${config.steps}
- Validation bits/byte: ${metrics.final.bitsPerByte.toFixed(3)}
- Unigram bits/byte: ${metrics.baselines.unigram.bitsPerByte.toFixed(3)}
- Bigram bits/byte: ${metrics.baselines.bigram.bitsPerByte.toFixed(3)}
- Improvement over unigram: ${(metrics.improvementOverUnigram * 100).toFixed(1)}%

The byte vocabulary avoids an external tokenizer. The edition mix, historical
English, and small corpus dominate the result. Generated samples are diagnostic
only. The memorization report checks exact copied spans and is not a privacy
guarantee.
`;
const artifactContents = new Map([
  ["metrics.json", `${JSON.stringify(metrics, null, 2)}\n`],
  [
    "history.jsonl",
    `${history.map((row) => JSON.stringify(row)).join("\n")}\n`,
  ],
  [
    "samples.jsonl",
    `${samples.map((row) => JSON.stringify(row)).join("\n")}\n`,
  ],
  ["checkpoint.json", `${JSON.stringify(checkpoint)}\n`],
  ["model-card.md", modelCard],
]);
run.artifacts = Object.fromEntries(
  [...artifactContents].map(([filename, contents]) => [
    filename,
    createHash("sha256").update(contents).digest("hex"),
  ]),
);

await mkdir(runRoot, { recursive: true });
await Promise.all([
  writeFile(path.join(runRoot, "run.json"), `${JSON.stringify(run, null, 2)}\n`),
  ...[...artifactContents].map(([filename, contents]) =>
    writeFile(path.join(runRoot, filename), contents),
  ),
]);

console.log(
  JSON.stringify({
    run: path.relative(root, runRoot),
    sha256: createHash("sha256")
      .update(JSON.stringify(metrics))
      .digest("hex"),
    metrics,
  }),
);

optimizer.dispose();
variables.forEach((variable) => variable.dispose());
