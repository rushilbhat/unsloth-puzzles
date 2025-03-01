# unsloth-puzzles

### Solution notebook
https://colab.research.google.com/github/rushilbhat/unsloth-puzzles/blob/main/Unsloth_Puzzles.ipynb

**Note**: This notebook was run on a T4 instance.

#### Instructions to Run the Notebook

- For each task, restart the session and run the first two cells (i.e., code to install Unsloth, Triton, Torch + helpful functions used throughout the entire notebook). This is because, in task C, I encountered the following error:
  ```
  /content/unsloth_compiled_cache/UnslothSFTTrainer.py in __init__(self, model, args, data_collator, train_dataset, eval_dataset, processing_class, compute_loss_func, compute_metrics, callbacks, optimizer_cls_and_kwargs, preprocess_logits_for_metrics, peft_config, formatting_func, **kwargs)
    893         float16 = dtype == torch.float16
    894         if float16 and use_bf16: raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
  --> 895         if not float16 and use_fp16: raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
    896         if not use_bf16 and not use_fp16:
    897             args.fp16 = float16
  
  TypeError: Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`, if I didn’t restart the session.
  ```
  
  Strangely, after restarting the session, the error disappears even though `model.config.dtype` is still `bfloat16` and `use_fp16` is `True`.
  
- When running task C, make sure to also run the cell defining the implemented Triton kernel from Task A.


### Tasks Attempted
A, C, E

### Role
Applying for the intern position.
