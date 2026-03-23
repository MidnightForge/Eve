"""
Eve Honeycomb Brain — distributed cell-based AI architecture.

Each cell controls one domain of Eve's function. The Cortex is the
central coordinator — always online via Claude API. Other cells
lazy-initialize and only run when their domain is needed.

         [emotion] [anima] [creative]
            [memory]─[CORTEX]─[voice]
         [vision]  [tools]  [reasoning]

Usage:
    from brain import brain
    response = await brain.process(message, context)
    status   = brain.status()
"""

from .manager import HoneycombBrainManager

# Singleton — one brain instance shared across the entire Eve backend
brain = HoneycombBrainManager()
