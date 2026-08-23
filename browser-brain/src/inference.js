export class TransformersLocalRuntime{
  constructor({modelId='ItsNotAILABS/HIM-native-onnx',localModelPath='/models/',wasmPath='/vendor/transformers/'}={}){
    this.modelId=modelId;this.localModelPath=localModelPath;this.wasmPath=wasmPath;this.generator=null;
  }
  async load(){
    const {env,pipeline}=await import('@huggingface/transformers');
    env.allowLocalModels=true;env.allowRemoteModels=false;env.localModelPath=this.localModelPath;
    env.backends.onnx.wasm.wasmPaths=this.wasmPath;
    this.generator=await pipeline('text-generation',this.modelId,{device:globalThis.navigator?.gpu?'webgpu':'wasm',dtype:'q8'});return this;
  }
  async generate(messages,{maxNewTokens=256,temperature=.4}={}){
    if(!this.generator)throw new Error('local model is not loaded');
    const output=await this.generator(messages,{max_new_tokens:maxNewTokens,temperature,do_sample:temperature>0});
    return {text:output[0]?.generated_text??'',engine:'transformers.js',model:this.modelId,remoteModelsAllowed:false};
  }
}
