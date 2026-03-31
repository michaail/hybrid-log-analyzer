FROM neo4j:5

# Default credentials — override at runtime with -e NEO4J_AUTH=neo4j/<password>
ENV NEO4J_AUTH=neo4j/password123

# Heap memory for larger graph imports
ENV NEO4J_server_memory_heap_initial__size=512m
ENV NEO4J_server_memory_heap_max__size=2G
ENV NEO4J_server_memory_pagecache_size=1G

# Allow connections from any host (required inside Docker)
ENV NEO4J_dbms_default__listen__address=0.0.0.0

EXPOSE 7474 7687

VOLUME ["/data"]