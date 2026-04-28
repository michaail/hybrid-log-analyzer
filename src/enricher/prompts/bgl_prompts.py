from langchain_core.prompts import ChatPromptTemplate

from .system import get_system_prompt

BGL_PROMPT_CORPUS = ChatPromptTemplate.from_messages(
  [
    get_system_prompt(),
    (
      "human",
      """
      # BGL Log Dataset Overview:
      - **System Description:** BGL (Blue Gene/L) is a massively parallel supercomputer architecture 
        designed by IBM. The system consists of tens of thousands of compute nodes, 
        I/O nodes, and specialized networking hardware (like a 3D Torus network). 
        Logs are generated continuously across this distributed hardware hierarchy, 
        with components typically identified by physical location codes (e.g., Rack, 
        Midplane, Node Card, Compute Node).
      - **Definitions of Anomalies and Failure Modes:** In the BGL high-performance computing (HPC) environment, an anomaly represents 
        a hardware or software failure that disrupts computational jobs or system integrity. 
        Common failure modes include memory corruption (e.g., ECC parity errors), network 
        routing issues, switch failures, I/O bus errors, power/cooling threshold alerts, 
        and software-level kernel panics. System alerts are often tagged with severity 
        levels (INFO, WARNING, SEVERE, ERROR, FATAL).
      - **Labeling Rules Used in the Benchmark:** Unlike session-based systems, BGL logs are a continuous chronological stream of 
        events. In the raw dataset, system administrators manually labeled individual log 
        messages as either normal or anomalous based on known failure alerts. For anomaly 
        detection modeling (like in Loghub benchmarks), these logs are typically grouped 
        into time-sliding windows or fixed event-count windows. A window is labeled as 
        an anomaly if it contains at least one log message flagged as a failure event; 
        otherwise, it is considered normal.
      - **Structured Instruction:** “summarize the essential semantic information relevant to anomaly detection, focusing 
        on the severity of the action, the affected component type, and the potential impact 
        on system stability.”

      Enrich the following pre-processed log template (where dynamic hardware IDs, addresses, 
      and IPs have been masked) with additional information according to the given context:
      {log_template}
      """
    )
  ]
)
