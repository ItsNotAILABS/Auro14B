export class KnowledgeGraph{
  constructor(store){this.store=store}
  async assert(node){if(!node?.type||!node?.label)throw new Error('node type and label required');return this.store.put('graph',{id:node.id||`node:${node.type}:${node.label.toLowerCase()}`,...node,kind:'node'})}
  async relate(from,predicate,to,evidence=[]){if(!from||!predicate||!to)throw new Error('complete edge required');return this.store.put('graph',{id:`edge:${from}:${predicate}:${to}`,...{from,predicate,to,evidence},kind:'edge'})}
  async neighborhood(id){const rows=await this.store.getAll('graph');return rows.filter(x=>x.id===id||x.from===id||x.to===id)}
  async export(){const rows=await this.store.getAll('graph');return {schema:'him.knowledge_graph.v1',nodes:rows.filter(x=>x.kind==='node'),edges:rows.filter(x=>x.kind==='edge')}}
}
