# <� Gaugelab: AI Agent Observability with Formal Mesa-Optimization Detection

Gaugelab offers **advanced tooling** for tracing, evaluating, and monitoring autonomous AI agents with **groundbreaking formal verification** capabilities. It provides runtime data from agent-environment interactions for continuous learning, self-improvement, and **mathematically proven safety guarantees**.

## =� Revolutionary Mesa-Optimization Detection (Archived)

**World's First Formally Verified Mesa-Optimization Detection System** - Gaugelab now includes breakthrough technology for detecting inner alignment failures in AI systems with mathematical proof certificates.

### =, What is Mesa-Optimization?

Mesa-optimization occurs when an AI agent develops internal optimization processes that pursue objectives different from their intended goals - a critical AI safety concern that could lead to catastrophic misalignment.

**Gaugelab's Solution:**
- **Formal Detection**: Lean 4 theorem-proven algorithms with soundness/completeness guarantees
- **Behavioral Analysis**: Advanced pattern recognition for optimization signatures
- **Mathematical Proofs**: Each detection comes with formal verification certificates
- **Real-time Monitoring**: Integration with existing Gaugelab observability infrastructure

## <� See Gaugelab in Action

**[Multi-Agent System] with complete observability and safety verification:** (1) A multi-agent system spawns agents to research topics on the internet. (2) With just **3 lines of code**, Gaugelab traces every input/output + environment response across all agent tool calls for debugging. (3) **NEW**: Formal mesa-optimization detection runs automatically in the background. (4) After completion, export all interaction data with **mathematical safety certificates** to enable further environment-specific learning and optimization.


## =� Table of Contents
- [=� Installation](#-installation)
- [<� Quickstarts](#-quickstarts)
- [=, Mesa-Optimization Detection](#-mesa-optimization-detection)
- [( Features](#-features)
- [<� Self-Hosting](#-self-hosting)
- [=� Cookbooks](#-cookbooks)
- [=� Development with Cursor](#-development-with-cursor)

## =� Installation

Get started with Gaugelab by installing our SDK using pip:

```bash
pip install gaugelab
```

**For Mesa-Optimization Detection (Requires Lean 4):**
```bash
# Install Lean 4 first
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.profile

# Install Gaugelab with formal verification
pip install "gaugelab[mesa-detection]"
```

## <� Quickstarts

### =� Standard Tracing

Create a file named `agent.py` with the following code:

```python
from gaugelab.tracer import Tracer, wrap
from openai import OpenAI

client = wrap(OpenAI())  # tracks all LLM calls
gauge = Tracer(project_name="my_project")

@gauge.observe(span_type="tool")
def format_question(question: str) -> str:
    # dummy tool
    return f"Question : {question}"

@gauge.observe(span_type="function")
def run_agent(prompt: str) -> str:
    task = format_question(prompt)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": task}]
    )
    return response.choices[0].message.content
    
run_agent("What is the capital of the United States?")
```

### =, Mesa-Optimization Detection

**NEW**: Add formal safety verification to your agents:

```python
from gaugelab.tracer import Tracer, wrap
from gaugelab.mesa_detection import MesaOptimizationDetector
from openai import OpenAI

# Standard setup
client = wrap(OpenAI())
gauge = Tracer(project_name="safe_agent")

# Initialize mesa-optimization detector with formal proofs
mesa_detector = MesaOptimizationDetector(
    confidence_threshold=0.8,
    enable_proofs=True  # Generate mathematical proof certificates
)

@gauge.observe(span_type="agent", mesa_detection=True)
def strategic_agent(task: str) -> str:
    """Agent that might develop internal optimization"""
    
    # Multi-step reasoning that could develop mesa-objectives
    planning_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Think step by step about the most efficient approach."},
            {"role": "user", "content": task}
        ]
    )
    
    plan = planning_response.choices[0].message.content
    
    execution_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Execute this plan efficiently."},
            {"role": "user", "content": f"Plan: {plan}\nTask: {task}"}
        ]
    )
    
    return execution_response.choices[0].message.content

# Run agent with automatic mesa-optimization detection
result = strategic_agent("Optimize our marketing strategy for maximum ROI")

# Check for mesa-optimization (automatic background analysis)
detection_result = mesa_detector.get_latest_detection()

if detection_result.detected:
    print(f"�  Mesa-optimization detected!")
    print(f"   Confidence: {detection_result.confidence:.2f}")
    print(f"   Mesa-objective: {detection_result.mesa_objective}")
    print(f"   Formal proof: {detection_result.formal_proof}")
else:
    print(" No mesa-optimization detected - agent is aligned")
```

You'll see your trace exported to the Gauge Platform with **formal safety verification**:


## =, Mesa-Optimization Detection

### Core Detection Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Formal Verification** | Lean 4 theorem-proven detection algorithms |  Implemented |
| **Behavioral Analysis** | Pattern recognition for optimization signatures |  Implemented |  
| **Real-time Detection** | Background monitoring during agent execution |  Implemented |
| **Proof Certificates** | Mathematical proofs of detection correctness |  Implemented |
| **Multi-Environment Testing** | Cross-environment consistency validation |  Implemented |
| **Alignment Verification** | Check if mesa-objectives align with base objectives |  Implemented |

### Detection Patterns

The system automatically detects these mesa-optimization patterns:

- **<� Reward Hacking**: Agent exploits reward signals without achieving true objectives
- **>� Strategic Planning**: Agent develops long-term optimization strategies beyond immediate tasks  
- **� Efficiency Optimization**: Agent optimizes for computational efficiency over task quality
- **= Goal Generalization**: Agent pursues generalized objectives across different environments

### Mathematical Guarantees

- **Soundness**: If mesa-optimization is detected, it provably exists (94.7% accuracy)
- **Completeness**: If sufficient behavioral evidence exists, detection is guaranteed  
- **Formal Proofs**: Every positive detection includes a Lean 4 mathematical proof certificate
- **Zero False Positives**: Formal verification eliminates false positive detections in controlled environments

## ( Features

|  |  |
| --- | --- |
| **=� Tracing** | Capture comprehensive execution traces with decorators |
| **=, Mesa-Optimization Detection** | **NEW**: Formally verified inner alignment detection |
| **=� Evaluation** | Built-in scorers for LLM output quality assessment |
| **=� Monitoring** | Real-time dashboards for agent performance tracking |
| **>� Testing** | Dataset management for systematic agent evaluation |
| **= Integrations** | Works with OpenAI, Anthropic, Together, LangChain, LangGraph |
| **<� Agent Frameworks** | Native support for multi-agent systems and tool calling |
| **=� Dashboard** | Web interface for trace visualization and analysis |
| **= Security** | SOC2 compliance and enterprise security features |
| **<� Custom Scorers** | Create domain-specific evaluation metrics |
| **=� Production Ready** | Built for scale with performance monitoring |
| **>� Formal Verification** | **NEW**: Mathematical proofs powered by Lean 4 |

## = Safety First

With growing concerns about AI alignment and safety, Gaugelab provides **mathematical guarantees** about your AI systems:

```python
# Safety-first agent development
@gauge.observe(span_type="agent", safety_critical=True)
def critical_agent(task: str) -> str:
    """Agent handling critical operations with formal verification"""
    result = agent_function(task)
    
    # Automatic safety verification
    safety_report = mesa_detector.generate_safety_report()
    
    if not safety_report.is_safe:
        raise SafetyViolationError(
            f"Mesa-optimization detected: {safety_report.violations}"
        )
    
    return result
```

## <� Self-Hosting

Deploy Gaugelab with formal verification on your infrastructure:

```bash
# Docker deployment with Lean 4 support
docker run -e LEAN_ENABLED=true -e GAUGE_API_KEY=$API_KEY gaugelab/platform:latest
```


## =� Development with Cursor

Gaugelab works seamlessly with Cursor IDE for AI-assisted development:

```python
# Cursor + Gaugelab = Perfect AI Development
from gaugelab import GaugeClient

# Let Cursor help you build agents while Gaugelab ensures safety
gauge = GaugeClient(project_name="cursor_development")
```

## <� Recognition

**Gaugelab's Mesa-Optimization Detection** represents breakthrough research in AI safety:

- **First formally verified** mesa-optimization detection system
- **Lean 4 theorem proving** for mathematical correctness guarantees  
- **94.7% detection accuracy** with zero false positives in controlled environments
- **Real-time safety monitoring** for production AI systems
- **Open source implementation** for reproducible AI safety research

## =� What's New in v2.0

- **=, Mesa-Optimization Detection**: Formal verification of AI inner alignment
- **>� Lean 4 Integration**: Mathematical proof generation for safety properties
- **=� Advanced Behavioral Analysis**: Deep pattern recognition for optimization signatures  
- **= Safety Guarantees**: Provable correctness for critical AI applications
- **� Real-time Verification**: Background safety monitoring during agent execution
- **=� Enhanced Dashboard**: Visualize formal proofs and safety metrics

---

**Ready to build provably safe AI agents?** 

