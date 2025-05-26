Personalized Query Auto-Completion for Long and Short-Term Interests with Adaptive Detoxification Generation
====================================


![alt text](image/model.png)
<p align="center">Overall structure of the LaD.</p>



This is a PyTorch implementation for [Personalized Query Auto-Completion for Long and Short-Term Interests with Adaptive Detoxification Generation, KDD-2025].


  * [Data](#Data)
  * [Train](#Train)


Copying the following code to cite:

```text
@inproceedings{wang2025lad,
  title =  {Personalized Query Auto-Completion for Long and Short-Term Interests with Adaptive Detoxification Generation},
  author =  {Wang, Zhibo and Jiang, Xiaoze and Qin, Zhiheng and Yu, Enyun and Li, Han},
  year =  {2025},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25), August 3--7, 2025, Toronto, ON, Canada}
}
```

Data
----------------------
We have given 10 anonymized training data samples in `data.txt`. Each sample contains three columns: prefix, query, searched_query.

Train
----------------------
Because LaD has been applied to Kuaishou online, the code is company confidential. Therefore, we make the training loss and training pseudocode public in `main.py` to help researchers better understand and reproduce our paper. 