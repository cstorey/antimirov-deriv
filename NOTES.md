# Exciting things to do before the heat death of the universe

  * Look at using using bitsets to represent the active States to emulate parallel NFA evaluation.  Can use matrix multiplication to do graph traversal, so BC. Valid traversals depend on input, have Char -> BitMatrix, where BitMatrix ~= Set usize -> Set usize.
    * Need to explore methods for efficient bit matrix multiplication, Four Russians, &c.
  * Actually understand the Thompson NFA construction.
    * Compare with re2.

# Groups / submatch / actions
 * Each transition ends up being annotated with a function, which are then composed along the derivattion tree. Perhaps the most "obvious" example is as a set of shift/reduce instructions for a forth-style stack machine.
 * This looks a lot like Parsing with Derivatives.
