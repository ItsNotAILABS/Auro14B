export type AgentEnv = { PRODUCT_VERSION: string };

type StoredEvent = {
  id: string;
  kind: string;
  payload: unknown;
  createdAt: string;
};

abstract class CoordinationUnit {
  protected readonly state: DurableObjectState;
  protected readonly env: AgentEnv;

  constructor(state: DurableObjectState, env: AgentEnv) {
    this.state = state;
    this.env = env;
    this.state.storage