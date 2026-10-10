# Multimodal AI, Chart Reading and Evidence

## What "Multimodal" Means

A multimodal system accepts and reasons over more than one input modality —
typically text and images, sometimes audio or video. The distinguishing feature
is not that a model can accept an image, but that the system decides how to
combine modalities: when to read text out of the image, when to interpret the
image visually, and when to ask the user to disambiguate.

## Interpreting a Machine Learning Graph

When reading a training curve:

- The **x-axis** is usually epochs or training steps.
- The **y-axis** is usually loss or accuracy.
- The **training curve** (often blue) is the loss on the training set.
- The **validation curve** (often orange/red) is the loss on the held-out set.

A typical healthy run shows both curves falling. Overfitting begins where the
validation curve flattens and then rises while the training curve keeps falling.
The widening vertical gap between the two curves is called the generalisation
gap. Answering "why does validation loss increase?" requires pointing at this
gap and at the epoch where the curves first diverge.

## Why Validation Loss Increases

Validation loss increases because the model has started fitting noise specific
to the training set. Its weights become tuned to patterns that do not exist in
unseen data, so its predictions on the validation set get worse even as training
performance improves. Common contributing factors are a small training set, a
high-capacity model, too many epochs, and no regularisation.

## Evidence and Uncertainty in Visual Question Answering

A trustworthy assistant distinguishes what it can see from what it assumes:

- If an image is blurry, a count is a *lower bound*, not an exact answer. The
  honest reply is "I can clearly identify three people; part of the image is
  obscured so the total is uncertain."
- Text that cannot be read should be reported as illegible, not guessed.
- Numeric values on a chart should be quoted only when the axis labels support
  them. Otherwise the assistant must say the value is unclear.

## Ambiguity Handling

A request such as "check this image" does not specify a task. The correct
behaviour is to ask what kind of analysis is wanted — a description, text
extraction, or a specific visual detail — rather than picking one silently and
presenting its output as if it were what the user asked for.

## Retrieval-Augmented Answers

For questions that require external facts, the assistant retrieves supporting
passages from a trusted corpus and grounds its answer in them. If retrieval
returns nothing relevant, the assistant should say the answer is not supported
rather than producing an unsupported claim.
