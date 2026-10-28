// Training and the webcam call the same extractor; no Python reimplementation.
import {readFile,writeFile} from 'node:fs/promises';
import {extract,FEATURE_VERSION} from '../public/features.js';
const [input,output]=process.argv.slice(2);
if(!input||!output)throw new Error('Usage: node scripts/extract_features.mjs input.json output.json');
const samples=JSON.parse(await readFile(input,'utf8'));
const rows=samples.map(sample=>({
  ...sample.metadata,
  features:extract(sample.face,sample.hand,sample.pose,sample.width,sample.height)
}));
await writeFile(output,JSON.stringify({featureVersion:FEATURE_VERSION,rows}));
console.log(`Extracted ${rows.length} samples with feature schema ${FEATURE_VERSION}`);
