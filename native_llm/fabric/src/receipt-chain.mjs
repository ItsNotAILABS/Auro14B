import { createHash, randomUUID } from 'node:crypto';

const ZERO_HASH = '0'.repeat(64);

export function createReceiptChain({ limit = 500 } = {}) {
  let head = ZERO_HASH;
  const receipts = [];

  return {
    append(kind, value) {
      const receipt = {
        id: randomUUID(),
        ts: Date.now(),
        kind,
        value,
        previous_hash: head,
      };
      receipt.hash = createHash('sha256')
        .update(JSON.stringify(receipt))
        .digest('hex');
      head = receipt.hash;
      receipts.unshift(receipt);
      if (receipts.length > limit) receipts.length = limit;
      return receipt;
    },
    history() {
      return { head, receipts: [...receipts] };
    },
    verify() {
      let expectedPrevious = ZERO_HASH;
      for (const receipt of [...receipts].reverse()) {
        if (receipt.previous_hash !== expectedPrevious) return false;
        const { hash, ...unsigned } = receipt;
        const expectedHash = createHash('sha256')
          .update(JSON.stringify(unsigned))
          .digest('hex');
        if (hash !== expectedHash) return false;
        expectedPrevious = hash;
      }
      return expectedPrevious === head;
    },
  };
}
