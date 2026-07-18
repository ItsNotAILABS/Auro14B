export class TransformersLocalRuntime{
  constructor({modelId='ItsNotAILABS/HIM-native-onnx',localModelPath='/models/',wasmPath='/vendor/transformers/',nativeModelPath='/models/HIM-native-v0/'}={}){
    this.modelId=modelId;this.localModelPath=localModelPath;this.wasmPath=wasmPath;this.nativeModelPath=nativeModelPath;this.generator=null;this.native=null;
  }
  async load(){
    const {env,pipeline}=await import('@huggingface/transformers');
    env.allowLocalModels=true;env.allowRemoteModels=false;env.localModelPath=this.localModelPath;
    env.backends.onnx.wasm.wasmPaths=this.wasmPath;
    try{this.generator=await pipeline('text-generation',this.modelId,{device:globalThis.navigator?.gpu?'webgpu':'wasm',dtype:'q8'});this.engine='transformers.js'}
    catch(error){const {NativeBrowserRuntime}=await import('./native-inference.js');this.native=await new NativeBrowserRuntime({modelPath:this.nativeModelPath}).load();this.engine='him-native-browser';this.loadWarning=error.message}return this;
  }
  async generate(messages,{maxNewTokens=256,temperature=.4}={}){
    if(this.native)return this.native.generate(messages,{maxNewTokens,temperature});
    if(!this.generator)throw new Error('local model is not loaded');
    const output=await this.generator(messages,{max_new_tokens:maxNewTokens,temperature,do_sample:temperature>0});
    return {text:output[0]?.generated_text??'',engine:'transformers.js',model:this.modelId,remoteModelsAllowed:false};
  }
}
