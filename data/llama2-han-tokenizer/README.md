# LLaMA 2 13B Tokenizer - Chinese Extending Version

这是 LLaMA 2 Tokenizer 的汉字扩充版本， 包含了国家语委在 2013 年发布的《通用规范汉字表》 中的全部 3,000 个一级汉字，以及二级汉字、三级汉字的一部分子集。

我们认为在汉语中使用字而非词作为最小的语言单位是更加合理的，并且我们内部的实验也发现基于字的 BPE Tokenizer 所训练的 LLM 通常具有更强的 NLU 性能。
