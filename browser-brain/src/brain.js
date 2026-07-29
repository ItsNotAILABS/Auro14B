import {BrainStore} from './store.js';
import {SecurityMonitor} from './security.js';
import {KnowledgeGraph} from './graph.js';
import {TransformersLocalRuntime} from './inference.js';
import {ResearchCouncil} from './research.js';
import {DocumentEngine} from './documents.js';
import {MemoryEngine} from './memory.js';
import {ComputeMesh} from './mesh.js';
import {WorkflowEngine} from './workflow.js';
import {PythonEngineBridge} from './python-bridge.js';
import {PythonWasmFluidizer} from './python-wasm-fluidizer.js';

export class EmbeddedBrain{
  constructor(options={}){
    this.store=new BrainStore(options.storeName);
    this.security=new SecurityMonitor(this.store,options.security);
    this.graph=new KnowledgeGraph(this.store);
    this.memory=new MemoryEngine(this.store);
    this.inference=new TransformersLocalRuntime(options.inference);
    this.documents=new DocumentEngine();
    this.mesh=new ComputeMesh(options.mesh);
    this.python=new PythonEngineBridge(options.python);
    this.fluidizer=new PythonWasmFluidizer(options.fluidizer);
    this.workflows=new WorkflowEngine(this.store);
    this.research=new ResearchCouncil({store:this.store,graph:this.graph,inference:this.inference,security:this.security});
    this.state={phase:'created',cycles:0,lastError:null};
  }
  async awaken(){
    await this.store.open();await this.inference.load();
    this.mesh
      .register('inference',x=>this.inference.generate(x.messages,x.options))
      .register('recall',x=>this.memory.recall(x.query,x.options))
      .register('triad',x=>this.python.triad(x.message,{context:x.context||'',fluidize:false}))
      .register('fluidize',x=>this.fluidizer.fluidize(x.report))
      .start();
    this.workflows
      .register('think',x=>this.mesh.submit('inference',{messages:[{role:'user',content:x.message}],options:x.options}))
      .register('triad-think',x=>this.think(x.message,{triad:true}))
      .register('recall',x=>this.memory.recall(x.query))
      .register('python-capability',x=>this.python.capability(x.name,x.arguments,x.approved));
    this.state.phase='ready';return this.snapshot();
  }
  async think(message,{triad=false}={}){
    const check=this.security.inspectText(message);await this.security.record({operation:triad?'triad-think':'think',...check});if(!check.allowed)throw new Error('input denied by security monitor');
    const relevant=await this.memory.recall(message);
    let result;
    if(triad){
      const context=JSON.stringify(relevant);
      const raw=await this.python.triad(message,{context,fluidize:false});
      const rendered=await this.fluidizer.fluidize(raw.structured_answer||{answer:raw.text||'',caveats:raw.blockers||[]});
      result={...raw,text:rendered.text,pythonWasmFluidizer:rendered};
    }else{
      result=await this.inference.generate([{role:'system',content:`You are HIM. Relevant local memory: ${JSON.stringify(relevant)}`},{role:'user',content:message}]);
    }
    await this.memory.remember({kind:'conversation',text:`User: ${message}\nHIM: ${result.text}`,importance:.7});this.state.cycles++;
    return {...result,state:this.snapshot(),memoryUsed:relevant.map(x=>x.id)};
  }
  snapshot(){return {...this.state,model:this.inference.native?'HIM-native-v0':this.inference.modelId,engine:this.inference.engine||'not-loaded',computeMesh:this.mesh.snapshot(),fluidizer:this.fluidizer.status(),remoteModelsAllowed:false,loadWarning:this.inference.loadWarning||null}}
}
export * from './store.js';export * from './security.js';export * from './graph.js';export * from './research.js';export * from './documents.js';export * from './inference.js';export * from './memory.js';export * from './native-inference.js';export * from './mesh.js';export * from './workflow.js';export * from './python-bridge.js';export * from './python-wasm-fluidizer.js';
