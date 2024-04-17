# VQA架构
在VQA任务中，通常会涉及以下几个主要组件：

- 视觉提取网络（Visual Extraction Network）：
这个网络部分通常使用预训练的卷积神经网络（CNN）如ResNet、VGG或更高级的视觉Transformer模型来处理输入的图像，并提取图像的特征表示。这些特征能够捕捉图像中的重要视觉信息，如物体的形状、颜色和空间关系等。

- 文本提取网络（Textual Extraction Network）：
文本部分一般使用自然语言处理模型，如LSTM、GRU或Transformer-based模型（如BERT、GPT等），来处理问题文本。这些模型能够理解和编码问题的语义内容，包括关键词和语法结构。

- 融合网络（Fusion Network）：
在提取了视觉和文本特征之后，融合网络将这些特征整合到一起，创建一个联合的特征表示，这对于理解问题并将其与视觉内容联系起来是必要的。融合策略可以简单如拼接（concatenation）或复杂如注意力机制（attention mechanisms）、图神经网络（GNNs）等，以更动态地结合视觉和文本信息。

- 答案网络（Answer Network）：
这是最终的分类或回归网络，用于生成或选择答案。根据任务的不同，答案可能是开放式的（需要生成文本答案）或封闭式的（从预设的答案列表中选择）。答案网络通常是一两层的全连接网络，输出层的设计依赖于答案的类型（例如，使用softmax输出层来处理多类分类任务）。


# Prototype

## Shared Cross-modal Prototype Matrix

实现了Shared Cross-modal Prototype Matrix的功能的关键方法包括：

- MultiThreadMemory 类的 forward 方法：这个方法处理跨模态信息的查询和响应，它使用了一个多头注意力机制来处理query, key, 和 value。它通过调用 memory_querying_responding 函数实现了跨模态原型的选择和相应的交互，这符合跨模态原型矩阵（Shared Cross-modal Prototype Matrix）的概念，允许模型在文本和视觉特征间进行交互和信息融合。

- _prepare_feature_forward 方法：在这个方法中，query_matrix 和 cmn_masks 的生成与管理涉及了跨模态原型矩阵的动态使用。该方法根据输入标签构建查询矩阵 query_matrix，并生成与之对应的掩码 cmn_masks，这些矩阵和掩码为跨模态交互提供了必要的基础设施。

- EncoderDecoder 类的 __init__ 方法中，跨模态原型矩阵（prototypes）的初始化：该方法中的 protypes 初始化涉及到从预训练模型加载的原型数据，这些原型能够为跨模态学习提供先验的知识。

- Transformer:
这个类定义了整个Transformer模型的结构，包括编码器（encoder）和解码器（decoder）。在 decode 方法中，它利用 memory_querying_responding 函数来处理文本特征（即目标序列）和共享跨模态原型矩阵之间的交互。这样，模型可以在生成报告的过程中融合视觉和文本的信息。

这些方法共同支撑了跨模态原型矩阵的功能，使模型能够在视觉和文本数据之间建立更有效的信息桥梁。