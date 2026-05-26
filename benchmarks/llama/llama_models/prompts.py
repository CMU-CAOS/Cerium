"""Shared prompt definitions for the unpacked and packed-input runners."""

PROMPTS = (

### Data Sample 0
"""Choose A, B, C or D
What particle has a negative charge?
A) Proton
B) Neutron
C) Electron
D) Photon
Answer: """
,

### Data Sample 1
"""Choose A, B, C or D
Which of these elements is a noble gas?
A) Oxygen
B) Neon
C) Hydrogen
D) Carbon
Answer: """
,

### Data Sample 2
"""Choose A, B, C or D
Modern computer processors include small, fast memory structures known as caches. These caches store copies of recently used data so that the processor can access them much more quickly than if it had to retrieve them from main memory. From a performance perspective, what is the main purpose of including cache memory in processor design?
A. To eliminate the need for RAM
B. To increase disk capacity
C. To encrypt frequently accessed data
D. To reduce the average latency of memory accesses
Answer: """

,

### Data Sample 3
"""Choose A, B, C, D, E, F, G, H, I or J
Which is the capital city of France?
A. Berlin
B. Madrid
C. Rome
D. Lisbon
E. Vienna
F. Paris
G. Brussels
H. Amsterdam
I. Zurich
J. Prague
Answer: """

,

### Data Sample 4
"""Choose A, B, C, D, E, F, G, H, I or J
A machine learning engineer notices that a model achieves nearly perfect accuracy on the training set but performs substantially worse on new evaluation data collected from the same distribution. Which phenomenon best explains this behavior?
A. Underfitting
B. Quantization
C. Overfitting
D. Regularization
E. Normalization
F. Data augmentation
G. Gradient clipping
H. Tokenization
I. Calibration
J. Distillation
Answer: """

)


def get_prompt(data_sample: int) -> str:
    """Return the prompt for a zero-based data-sample index."""
    if not 0 <= data_sample < len(PROMPTS):
        raise ValueError(
            f"data_sample must be between 0 and {len(PROMPTS) - 1}, got {data_sample}"
        )
    return PROMPTS[data_sample]

