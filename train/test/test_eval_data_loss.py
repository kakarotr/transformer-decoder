import math

import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast

from models.config import TransformerConfig
from models.liger.causal_lm import CausalLanguageModel

sample = """
AB小报No.625 | 14亿支票未被存入，CRA喊7万多人收钱；快来了解航班延误取消的赔偿知识；联邦正在起草临时牙科保健计划

图源：Global News

在旅行需求激增的情况下，今年夏天航班延误和取消已成为常态，因此客户有权知道他们是否可以根据联邦法规获得航空公司的赔偿。

《航空乘客保护条例》（Air Passenger Protection Regulations）规定，如果机票持有人晚于三个小时到达目的地，或者如果他们的航班被取消且与安全问题无关或超出航空公司的控制范围，那么他们有权获得补偿。

一些航空公司因其对中断的解释而受到批评，包括加航和西捷都将员工短缺视为安全问题。加拿大交通局（CTA）则对以“员工短缺”为由提出异议，认为这符合补偿规则。

CTA在7月8日的一项裁决称，航空公司将员工短缺作为
产量不同，价格会不同，系统内的配套设备不同，价格也会有很大差距，甚至一个小配件不同都会带来某些差别，但是很多时候，客户的需求却是非常不清晰，例如我们经常收到如下询盘：

请报一套方便面生产线的价格，目的港为旧金山。

看到这样的报价我们会很为难，因为方便面生产线有太多种了，仅仅有这样的条件严格意义来讲我们是根本没法报价的。

很多人会讲，这样的客户是真正的客户吗？只说一句，我成交过很多个。

很多人会回邮件问客户，您的具体需求是什么呢？可惜事实证明，大部分邮件都是石沉大海，因为很多时候客户并没有完全确定自己所需要设备的产量，更多的是看厂家的报价与配置，所以，他们不愿意或者没法回答，这又是一种流失。

所以尴尬了，直接回复，怕信息不匹配，客户流失；回复邮件询问具体信息，客户又是一部分流失！

怎么做呢？

第一步还是背景调查，然后把调查出来的东西用在回复中，同时，寻找一个说的过去的理由，来推荐产品。说得过去就是有理有据，例如，印度人来买，我们可以推荐半自动，因为人工不算贵，他们预算也不会很高，半自动会比较给对方省预算，当然，如果我们发现对方规模很大，实力雄厚，那么我们就应该推荐一些高级配置，并且说明高级配置带来的好处。当然，不要忘了，说明一下，低级配置我们是有的，例如，哪些地方较为低级，价格大体为多少。最后问上一句，不知道您是否有比较详细的需求？哪怕只有占地面积，投资金额之类，我们也可以提供solution。

这个是真的提高了回复率。

所以，面对型号款式未定的客户的回复公式是：

背景调查+款式推荐+推荐理由（卖点，根据背景调查匹配）+精确信息获取。

还有很多种产品，没法一一举例，大家如果真的想做好外贸，下面这个拆解大家一定要会：

首先，穷举你能收到询盘的所有可能性

然后，背景调查

然后根据背景调查的资料，已经你对你产品的了解，针对性的提供solution给客户。这个过程就是上面绝大部分篇幅所提到的分析。

可以说，如果可以做到这一步拆解，基本上可以极大的提升第一封邮件的回复率。

邮件回复并不是单单给客户报价，更重要的是研究客户的需求，提供解决方案，没有真正的解决方案的邮件，没有回复你也是意料之中的了，所以学会询盘回复的正确打开方式了吗？

外贸牛一站式外贸整合营销领先品牌，拥有专业的SEO和SEM技术，用数据驱动营销，用社交提高转换，让您的外贸推广流量无忧，询盘翻番，转化有道。<|im_end|>美育到底能给孩子带来什么？这些你需要知道！

一朵云在大人眼中
是一团水汽
在孩子眼中
是马儿、是大象、是城堡、是亭台楼阁
乃至一整个奇异世界
而美育就是
回归对孩子生命直觉的引导
用美来温润孩子的眼睛和心灵

美育是什么？

美育是通过一定的媒介让孩子看到美，听见美，触摸美，感受美，从而成为一个自由、敏感而富有创造力的人，化身生活的艺术家。

正如蔡元培先生所说“美育的目的在于陶冶人的感情，认识美丑，培养高尚的兴趣和积极进取的人生态度”。

它不仅仅指视觉美学的培养，还包括想象力、创造力及审美综合能力，是感性的、情感的、心灵的教育，是价值观、人生观形成的有效途径。

想象所及的地方，鲨鱼在飞翔，花朵在歌唱，石头也变得柔软。

美育是一个潜移默化的过程，也是一个储存能量的过程，它不是功利的，它是在让孩子慢慢学会欣赏生活的美好。

「自然」育人，「城市」育人，「艺术」育人，从空间环境延展到精神环境，美学的课堂无处不在。

美育到底能给孩子带来什么？

在很长一段时间，美育在中国只是饭后茶余可有可无的东西。

中国的大多数家长因为升学压力，让他们只顾孩子的学习，只要是与升学考试无关的东西都要去掉，没有艺术、没有交友、没有自由、甚至是没有体育……

数数、背九九表，背英语单词、背26个字母、甚至背元素周期表，这些充斥着中国孩子的童年，这样的教育：是赢在起跑线，输在终点站。

而接受过美育的孩子，更早地拥有独立思考的能力，拥有感知世界的能力，拥有自我交流的能力，在小编看来，这一切相加，才是真的赢在起跑线上。

为什么要对孩子进行美育

一个人想要获得一生的幸福，不仅要拥有获得幸福的生活条件，还需拥有体验幸福感受的能力与素质。因为体验幸福，是需要素质的。而对孩子进行美育，就是为了培养孩子的这种素质。

就美育而言，学到的技能，受益一时，将艺术融于生活，练就的是思维，影响一世。

真正懂艺术的人，能够把生活过得更细致，他们是这个快节奏时代里，最容易体会到细碎美好的那一小部分人。

学会尊重孩子的兴趣爱好

美国教育家斯宾塞说过：“身为父母，千万不能太看重孩子的考试分数，而应该注重孩子思维能力、学习方法的培养，尽量留住孩子最宝贵的兴趣与好奇心。绝对不能用考试分数去判断一个孩子的优劣，更不能让孩子有以此为荣辱的意识。

孩子的发展应当是全面的。父母培养孩子首先要发现孩子的特长与爱好，不能使每一个孩子都变成一个学习的机器 ，而应当使他得到全面的发展。只要孩子的兴趣爱好不是负面的，我们就要加以鼓励和保护，并且要尊重孩子的兴趣。

艺术是感知生命的自由

孩子的精神世界构建于对细小事物的认知，他们徜徉于艺术世界之中，会获得更多的灵感和启发，除了用艺术的眼光看待周遭的事物外，更会无意识的用歌声、舞蹈、画笔、文字去描绘他们所感知的世界。

让孩子接触艺术教育，不只是为了让他们学习，而是在于借助艺术这种手段，将孩子自由、纯真、富有创造力的天性进一步释放，让孩子去感知生命的自由。

艺术教育，应该是条彩色的通道，为人父母，我们应做的就是陪伴孩子穿越其间，最终抵达孩子内心中最通透、最快乐的那片丰饶之地，让他们愿意倾听自己的内心，珍惜生活的美好，也表达他们对生活的认知。<|im_end|>
"""

if __name__ == "__main__":
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("artifacts/base")
    with open("artifacts/config.json", mode="r") as f:
        config = TransformerConfig.model_validate_json(f.read())

    model = CausalLanguageModel(config)
    state_dict = load_file("artifacts/model.safetensors")
    device = torch.device("cuda")
    model.load_state_dict(state_dict)
    model.to(device).eval()

    inputs = tokenizer(sample)
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    print(input_ids.shape)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = model(input_ids)
            weight = model.lm_head.weight.to(hidden_states.dtype)

            shift_h = hidden_states[:, :-1].contiguous()
            shift_l = input_ids[:, 1:].contiguous()
            flat_h = shift_h.view(-1, shift_h.size(-1))
            flat_l = shift_l.view(-1)

            liger_loss = LigerFusedLinearCrossEntropyLoss()(weight, flat_h, flat_l)

            logits = F.linear(flat_h, weight).float()
            std_loss = F.cross_entropy(logits, flat_l)

print(f"Liger loss: {liger_loss.item():.4f} ppl: {math.exp(liger_loss.item())}")
print(f"Standard CE: {std_loss.item():.4f} ppl: {math.exp(std_loss.item())}")
