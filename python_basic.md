Protocol理解与学习：Protocol（结构化子类型 / 鸭子类型协议，PEP 544）**，属于静态类型检查的特性：**不需要显式继承，只要类实现了协议里约定的方法签名，就自动满足这个协议**。

```
from typing import Protocol

class Retriever(Protocol):          
    def retrieve(self, query: str) -> list[str]: ...

class MyRetriever:
    def retrieve(self, query: str) -> list[str]:
        return ["文档片段A", "文档片段B"]

def search(r: Retriever, q: str) -> list[str]:
    return r.retrieve(q)

print(search(MyRetriever(), "年假制度")) 
```

其中search方法的第一个参数r的类型注解为Retriever,在编程时，IDE 里敲 `r.`，会自动补全 `.retrieve`；

可以看到虽然MyRetriever没有继承Retriever，但是可以将MyRetriever()实例传给search方法。
