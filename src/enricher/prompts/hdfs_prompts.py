from langchain_core.prompts import ChatPromptTemplate
from .system import get_system_prompt


HDFS_PROMPT_CORPUS = ChatPromptTemplate.from_messages(
  [
    get_system_prompt(),
    (
      "human",
      """
      # HDFS Log Dataset Overview:
      - **System Description:** 
        HDFS (Hadoop Distributed File System) is a highly fault-tolerant 
        distributed file system designed to store massive amounts of data across clusters 
        of commodity hardware. It operates on a primary-secondary architecture where 
        a central "NameNode" manages the file system metadata, and multiple "DataNodes" 
        store the actual data as split blocks.
      - **Definitions of Anomalies and Failure Modes:** 
        In HDFS operations, an anomaly is any 
        deviation from the standard data block lifecycle. Common failure modes include DataNode 
        disconnections, block write/read exceptions, missing or corrupted data blocks, 
        replication failures (e.g., under-replicated or over-replicated blocks), and network timeouts.
      - **Labeling Rules Used in the Benchmark:** 
        In standard HDFS benchmark datasets (like those from Loghub), log messages are first 
        parsed and grouped into sequences based on their unique block_id. A block's log sequence 
        is labeled as an anomaly if it contains events matching known failure modes or if the 
        lifecycle is interrupted prematurely (e.g., a block is allocated but never successfully written). 
        It is labeled as normal if the sequence perfectly follows the expected operations 
        (allocation → writing → successful replication → completion or normal deletion).
      - **Structured Instruction:** 
        “summarize the essential semantic information relevant to anomaly detection.”

      Enrich the following log template with additional information according to the given context:
      {log_template}
      """
    )
  ]
)
