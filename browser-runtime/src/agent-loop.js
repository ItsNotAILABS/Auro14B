export class GovernedChromeAgentLoop {
  constructor({ infer, observe, proposeAction, executeAction, memory, policy }) {
    this.infer = infer;
    this.observe = observe;
    this.proposeAction = proposeAction;
    this.executeAction = executeAction;
    this.memory = memory;
    this.policy = policy;
  }

  async step(task) {
    const observation = await this.observe();
    const context = await this.memory.snapshot?.() ?? {};
    const modelOutput = await this.infer({ task, observation, context });
    const proposal = await this.proposeAction(modelOutput, observation);
    const decision = await this.policy.evaluate(proposal, { task, observation, context });

    const receipt = {
      schema: 'auro.chrome_agent_step.v1',
      task,
      observation,
      proposal,
      decision,
      executed: false,
      result: null,
    };

    if (!decision.allowed) {
      await this.memory.append?.({ type: 'policy_denial', receipt });
      return receipt;
    }
    if (decision.approvalRequired && !decision.approved) {
      await this.memory.append?.({ type: 'approval_required', receipt });
      return receipt;
    }

    receipt.result = await this.executeAction(proposal);
    receipt.executed = true;
    await this.memory.append?.({ type: 'execution', receipt });
    return receipt;
  }

  async run(task, { maxSteps = 12, stop = (receipt) => Boolean(receipt.result?.done) } = {}) {
    const receipts = [];
    for (let index = 0; index < maxSteps; index += 1) {
      const receipt = await this.step(task);
      receipts.push(receipt);
      if (!receipt.executed || stop(receipt)) break;
    }
    return { schema: 'auro.chrome_agent_run.v1', task, receipts };
  }
}
