export class WorkflowEngine{
  constructor(store){this.store=store;this.handlers=new Map()}
  register(kind,handler){this.handlers.set(kind,handler);return this}
  async create({name,tasks}){if(!name||!Array.isArray(tasks)||!tasks.length)throw new Error('workflow name and tasks required');const id=crypto.randomUUID();return this.store.put('research',{id:`workflow:${id}`,kind:'workflow',name,status:'queued',tasks:tasks.map((t,i)=>({id:t.id||`task-${i+1}`,kind:t.kind,input:t.input||{},dependsOn:t.dependsOn||[],status:'queued'})),createdAt:new Date().toISOString()})}
  async run(workflow){workflow.status='running';for(const task of workflow.tasks){if(task.status==='completed')continue;const blocked=task.dependsOn.some(id=>workflow.tasks.find(x=>x.id===id)?.status!=='completed');if(blocked)continue;const handler=this.handlers.get(task.kind);if(!handler){task.status='blocked';task.error=`no handler: ${task.kind}`;continue}task.status='running';try{task.output=await handler(task.input,workflow);task.status='completed'}catch(error){task.status='failed';task.error=error.message;workflow.status='failed';break}await this.store.put('research',workflow)}if(workflow.tasks.every(t=>t.status==='completed'))workflow.status='completed';await this.store.put('research',workflow);return workflow}
  async list(){return (await this.store.getAll('research')).filter(x=>x.kind==='workflow')}
}
