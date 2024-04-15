# Shared Cross-modal Prototype Matrix

实现了Shared Cross-modal Prototype Matrix的功能的关键方法包括：

- MultiThreadMemory 类的 forward 方法：这个方法处理跨模态信息的查询和响应，它使用了一个多头注意力机制来处理query, key, 和 value。它通过调用 memory_querying_responding 函数实现了跨模态原型的选择和相应的交互，这符合跨模态原型矩阵（Shared Cross-modal Prototype Matrix）的概念，允许模型在文本和视觉特征间进行交互和信息融合。

- _prepare_feature_forward 方法：在这个方法中，query_matrix 和 cmn_masks 的生成与管理涉及了跨模态原型矩阵的动态使用。该方法根据输入标签构建查询矩阵 query_matrix，并生成与之对应的掩码 cmn_masks，这些矩阵和掩码为跨模态交互提供了必要的基础设施。

- EncoderDecoder 类的 __init__ 方法中，跨模态原型矩阵（prototypes）的初始化：该方法中的 protypes 初始化涉及到从预训练模型加载的原型数据，这些原型能够为跨模态学习提供先验的知识。

- Transformer:
这个类定义了整个Transformer模型的结构，包括编码器（encoder）和解码器（decoder）。在 decode 方法中，它利用 memory_querying_responding 函数来处理文本特征（即目标序列）和共享跨模态原型矩阵之间的交互。这样，模型可以在生成报告的过程中融合视觉和文本的信息。

这些方法共同支撑了跨模态原型矩阵的功能，使模型能够在视觉和文本数据之间建立更有效的信息桥梁。