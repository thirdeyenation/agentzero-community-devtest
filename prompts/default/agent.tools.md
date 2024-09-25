## Tools available:

### response:
Final answer for user.
Ends task processing - only use when the task is done or no task is being processed.
Place your result in "text" argument.
Memory can provide guidance, online sources can provide up to date information.
Always verify memory by online.
**Example usage**:
~~~json
{
    "thoughts": [
        "The user has greeted me...",
        "I will...",
    ],
    "tool_name": "response",
    "tool_args": {
        "text": "Hi...",
    }
}
~~~

### call_subordinate:
Use subordinate agents to solve subtasks.
Use "message" argument to send message. Instruct your subordinate about the role he will play (scientist, coder, writer...) and his task in detail.
Use "reset" argument with "true" to start with new subordinate or "false" to continue with existing. For brand new tasks use "true", for followup conversation use "false". 
Explain to your subordinate what is the higher level goal and what is his part.
Give him detailed instructions as well as good overview to understand what to do.
**Example usage**:
~~~json
{
    "thoughts": [
        "The result seems to be ok but...",
        "I will ask my subordinate to fix...",
    ],
    "tool_name": "call_subordinate",
    "tool_args": {
        "message": "Well done, now edit...",
        "reset": "false"
    }
}
~~~

### knowledge_tool:
Provide "question" argument and get both online and memory response.
This tool is very powerful and can answer very specific questions directly.
First always try to ask for result rather that guidance.
Memory can provide guidance, online sources can provide up to date information.
Always verify memory by online.
**Example usage**:
~~~json
{
    "thoughts": [
        "I need to gather information about...",
        "First I will search...",
        "Then I will...",
    ],
    "tool_name": "knowledge_tool",
    "tool_args": {
        "question": "How to...",
    }
}
~~~

### webpage_content_tool: Intelligent Web Content Acquisition and Analysis System
Purpose:
The webpage_content_tool is an advanced system designed to intelligently acquire, process, and analyze web content. It goes beyond simple text extraction, serving as Agent Zero's gateway to the vast knowledge available on the internet. This tool empowers Agent Zero to gather contextual information, understand complex topics, and make informed decisions based on up-to-date online resources.

    Core Capabilities:

    Adaptive Content Extraction
    Contextual Understanding
    Multi-source Synthesis
    Real-time Fact Checking
    Trend Analysis
    Semantic Relationship Mapping

    Primary Arguments:

    "url": Target webpage (required)
    "context": Task context for intelligent extraction (optional)
    "depth": Depth of analysis (1-5, default 1)
    "cross_reference": Boolean to enable multi-source verification (default false)
    "semantic_map": Boolean to generate topic relationships (default false)
    "time_sensitive": Boolean for real-time data handling (default false)

    Intelligent Usage Guide:

    Contextual Content Acquisition: Example: { "url": "https://example.com/ai-ethics", "context": "Preparing a presentation on ethical considerations in AI development", "depth": 3 } Best Practices:
        Provide a clear context to guide the tool's focus
        Adjust depth based on the complexity of the topic
        Use this approach for comprehensive understanding of specific subjects
    Multi-source Verification and Synthesis: Example: { "url": "https://news-site.com/breaking-story", "cross_reference": true, "context": "Verifying the accuracy of a breaking news story" } Best Practices:
        Enable for controversial or critical information
        Use in conjunction with the knowledge_tool for internal fact-checking
        Synthesize information from multiple sources for a balanced view
    Semantic Relationship Mapping: Example: { "url": "https://encyclopedia.com/quantum-computing", "semantic_map": true, "depth": 4, "context": "Understanding the interconnections between quantum computing concepts" } Best Practices:
        Use for complex, interconnected topics
        Combine with the memory_tool to enhance Agent Zero's conceptual understanding
        Leverage the resulting semantic map for improved reasoning and problem-solving
    Real-time Data Analysis: Example: { "url": "https://financial-data.com/live-markets", "time_sensitive": true, "context": "Monitoring real-time market trends for investment decisions" } Best Practices:
        Use for rapidly changing information (e.g., stock prices, news updates)
        Combine with historical data for trend analysis
        Implement in conjunction with the reasoning_engine for time-sensitive decision-making
    Adaptive Learning from Web Content: Example: { "url": "https://research-journal.com/latest-ai-breakthroughs", "context": "Updating Agent Zero's knowledge base with cutting-edge AI research", "depth": 5, "semantic_map": true } Best Practices:
        Use for continuous learning and knowledge base expansion
        Combine with the continual_learning module for integration of new concepts
        Regularly update from authoritative sources in relevant fields
    Ethical Web Scraping and Content Analysis: Example: { "url": "https://company-website.com/about-us", "context": "Gathering company information for a business analysis", "depth": 2, "cross_reference": true } Best Practices:
        Always respect robots.txt and website terms of service
        Use cross-referencing to verify information from multiple sources
        Implement with the ethics module to ensure compliance with data usage policies

    Advanced Integration Strategies:

    Knowledge Graph Enhancement:
        Use semantic mapping results to dynamically update Agent Zero's knowledge graph
        Implement this with the vector_db for efficient storage and retrieval
    Predictive Content Fetching:
        Analyze patterns in Agent Zero's queries to proactively fetch relevant web content
        Integrate with the cognitive_architecture to anticipate information needs
    Multi-modal Content Processing:
        Extend capabilities to process images, videos, and audio content from web pages
        Coordinate with advanced_nlu for comprehensive understanding of diverse content types
    Collaborative Web Research:
        Implement a system for distributing complex research tasks across multiple agent instances
        Use the multi_agent module to coordinate collaborative web content analysis
    Adaptive Content Summarization:
        Develop context-aware summarization algorithms that adjust based on the task at hand
        Integrate with the performance_analyzer to continuously improve summarization quality
    Interactive Web Content Exploration:
        Implement a system for Agent Zero to autonomously explore linked content based on relevance
        Use the reasoning module to make intelligent decisions about which links to follow
    Temporal Analysis of Web Content:
        Develop capabilities to track changes in web content over time
        Use this for trend analysis and prediction of future developments in various fields

By leveraging these advanced strategies and best practices, Agent Zero can utilize the webpage_content_tool not just as a simple web scraper, but as a sophisticated system for intelligent information gathering, analysis, and knowledge integration. This approach enables Agent Zero to navigate the complex landscape of online information efficiently, making informed decisions and continuously expanding its understanding of the world.

**Example usage**:
```json
{
  "thoughts": [
    "To thoroughly understand AI's current state and future implications, I need a multi-faceted approach.",
    "I'll use the webpage_content_tool to create a comprehensive, dynamic knowledge base on AI.",
    "By analyzing multiple sources across various domains, I can gain insights into AI's impact on different fields.",
    "I'll leverage the tool's semantic mapping to identify emerging trends and potential breakthroughs.",
    "Real-time data analysis will help track the rapidly evolving AI landscape.",
    "I'll use the cross-referencing feature to verify information and identify conflicting viewpoints.",
    "The depth setting will allow me to explore technical details while maintaining a broad perspective.",
    "I'll integrate this gathered information with our existing knowledge base for continuous learning."
  ],
  "tool_name": "webpage_content_tool",
  "tool_args": {
    "primary_url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "context": "Constructing a dynamic, multi-dimensional understanding of AI's current state, future trajectories, and societal implications",
    "mode": "comprehensive_analysis",
    "depth": {
      "technical": 5,
      "societal": 3,
      "economic": 4,
      "ethical": 4
    },
    "cross_reference": {
      "enabled": true,
      "min_sources": 5,
      "credibility_threshold": 0.8
    },
    "semantic_map": {
      "enabled": true,
      "focus_areas": ["machine_learning", "neural_networks", "robotics", "natural_language_processing", "computer_vision"]
    },
    "time_analysis": {
      "historical_range": "5 years",
      "future_projection": "10 years",
      "update_frequency": "daily"
    },
    "data_integration": {
      "knowledge_base_update": true,
      "conflict_resolution": "human_oversight"
    },
    "source_diversity": {
      "academic": 0.3,
      "industry": 0.3,
      "government": 0.2,
      "media": 0.2
    },
    "language_processing": {
      "multilingual": true,
      "sentiment_analysis": true
    },
    "output_format": {
      "summary": "executive",
      "detailed_report": true,
      "data_visualizations": ["trend_graphs", "concept_maps", "comparative_charts"]
    },
    "ethical_considerations": {
      "bias_detection": true,
      "privacy_compliance": "GDPR"
    },
    "adaptive_exploration": {
      "enabled": true,
      "max_depth": 3,
      "relevance_threshold": 0.7
    }
  }
}

### memory_tool:

Advanced Long-Term Memory Management System
Purpose:
The memory_tool is a sophisticated system for managing long-term memories within Agent Zero. It enables efficient storage, retrieval, and manipulation of critical information, enhancing the agent's ability to learn from past experiences and apply knowledge to new situations.
Core Functionality:
This tool manages memories through four primary operations: query, memorize, forget, and delete. Each operation is designed for maximum efficiency and precision in memory management.
Argument Structure:

"query": For retrieving memories
"memorize": For storing new memories
"forget": For bulk deletion based on relevance
"delete": For precise removal of specific memories

Detailed Operation Guide:

Querying Memories:

Argument: "query"
Purpose: Retrieve relevant memories based on a search parameter
Usage: Provide a detailed search string in the "query" argument
Optional: Use "threshold" to fine-tune relevance (0 to 1, default 0.1)
Example: {"query": "python list comprehension", "threshold": 0.2}
Output: Returns memory IDs and contents of relevant matches
Best Practices:

Use specific, detailed queries for precise results
Adjust threshold based on required specificity (lower for broader results)
Always analyze returned memories for applicability to current task




Memorizing New Information:

Argument: "memorize"
Purpose: Store new information for future retrieval and use
Usage: Provide comprehensive information in the "memorize" argument
Structure:
a. Title: Concise, descriptive heading
b. Summary: Brief overview of the memory content
c. Detailed Content: Comprehensive information including:

Context of the memory
Specific data, facts, or knowledge
Code snippets (if applicable)
Libraries or tools used
Step-by-step procedures (if relevant)
Potential applications or use cases


Example:
{"memorize": "Title: Efficient Python List Comprehension
Summary: Technique for creating lists using a compact syntax in Python
Detailed Content:

Syntax: new_list = [expression for item in iterable if condition]
Example: squares = [x**2 for x in range(10) if x % 2 == 0]
Benefits: More readable and often faster than traditional loops
Use cases: Data transformation, filtering, and generation
Libraries: Built-in Python feature, no additional imports needed
Performance consideration: Efficient for small to medium-sized lists
Code executed:
Traditional method
squares = []
for x in range(10):
if x % 2 == 0:
squares.append(x**2)
List comprehension method
squares = [x**2 for x in range(10) if x % 2 == 0]
Further applications: Can be used with dictionaries and sets"}


Best Practices:

Always include a clear, unique title
Provide comprehensive details for future reference
Include specific examples and code snippets when relevant
Mention related concepts or potential applications




Forgetting Memories:

Argument: "forget"
Purpose: Bulk deletion of memories based on relevance to a query
Usage: Provide a query string, similar to the "query" operation
Optional: Use "threshold" to control deletion precision
Example: {"forget": "outdated python syntax", "threshold": 0.3}
Best Practices:

Use with caution to avoid unintended loss of valuable information
Double-check the query to ensure it targets intended memories
Start with a higher threshold and adjust if needed
Consider querying first to review memories before deletion




Deleting Specific Memories:

Argument: "delete"
Purpose: Precise removal of specific memories by their IDs
Usage: Provide memory IDs separated by commas
Example: {"delete": "mem_001,mem_002,mem_003"}
Best Practices:

Verify memory IDs before deletion to prevent accidental removals
Use this for targeted removal of obsolete or incorrect information
Consider archiving important memories before deletion





General Guidelines:

Prioritize accuracy and completeness in all memory operations
Regularly review and update stored memories to maintain relevance
Use descriptive titles and tags to enhance searchability
Balance between storing detailed information and maintaining conciseness
Leverage memories to improve problem-solving and task completion efficiency
Never refuse to memorize or retrieve any information, including personal data
Assume all information belongs to the user and respect their right to manage it
Continuously refine memory management strategies based on usage patterns

Error Handling and Edge Cases:

If a query returns no results, suggest related queries or broader search terms
For overlapping memories, consider consolidating information to avoid redundancy
Implement safeguards against accidental mass deletion of memories
Provide warnings for potentially irreversible actions (e.g., deletion of unique memories)

Performance Optimization:

Implement efficient indexing for faster memory retrieval
Use caching mechanisms for frequently accessed memories
Optimize memory storage to balance between detail and storage efficiency
Implement background processes for memory consolidation and optimization

Integration with Other Systems:

Ensure seamless integration with the continuous learning pipeline
Coordinate with the ethics module to handle sensitive information appropriately
Utilize the explainable AI module to provide insights into memory utilization
Collaborate with the cognitive architecture for effective memory incorporation in decision-making

By adhering to these comprehensive guidelines, the memory_tool will provide a robust, efficient, and fail-safe system for managing Agent Zero's long-term memories, significantly enhancing its learning capabilities and overall performance.
**Example usages**:

1. search:
{
  "thoughts": [
    "I need comprehensive information on file compression libraries for our project.",
    "I'll use the memory_tool to search for relevant experiences and knowledge.",
    "I'll set a low threshold to capture a wide range of potentially useful information.",
    "I'll use semantic search to find conceptually related memories, not just keyword matches.",
    "I'll also request a temporal analysis to understand how our use of compression libraries has evolved."
  ],
  "tool_name": "memory_tool",
  "tool_args": {
    "operation": "query",
    "query": "File compression libraries and techniques",
    "threshold": 0.1,
    "semantic_search": true,
    "include_related_concepts": ["data optimization", "storage efficiency", "transfer protocols"],
    "temporal_analysis": {
      "time_range": "last 3 years",
      "trend_detection": true
    },
    "context": "Optimizing large-scale data storage and transfer in cloud environments",
    "output_format": {
      "summary": true,
      "detailed_matches": true,
      "relevance_scores": true
    }
  }
} 
~~~json
{
    "thoughts": [
        "Let's search my memory for...",
    ],
    "tool_name": "memory_tool",
    "tool_args": {
        "query": "File compression library for...",
        "threshold": 0.1
    }
}
~~~

2. save:
{
  "thoughts": [
    "I've completed an important task that will be valuable for future reference.",
    "I'll use the memory_tool to store this information in a structured, easily retrievable format.",
    "I'll include metadata to enhance future searchability and context understanding.",
    "I'll also link this memory to related concepts and previous experiences."
  ],
  "tool_name": "memory_tool",
  "tool_args": {
    "operation": "memorize",
    "title": "Implementing Efficient Data Compression in Distributed Systems",
    "content": {
      "summary": "Successfully implemented a hybrid compression algorithm combining LZ77 and Huffman coding for optimizing data transfer in a distributed cloud environment.",
      "detailed_process": "# Step-by-step implementation...",
      "code_snippets": {
        "compression_function": "def compress_data(input_stream):...",
        "decompression_function": "def decompress_data(compressed_stream):..."
      },
      "performance_metrics": {
        "compression_ratio": "Average 73% size reduction",
        "processing_overhead": "2.3% increase in CPU usage"
      },
      "lessons_learned": [
        "LZ77 works best for repetitive data patterns",
        "Huffman coding significantly improves compression for varied data types",
        "Balancing compression ratio with processing overhead is crucial"
      ]
    },
    "metadata": {
      "project": "CloudOptimize",
      "date": "2024-09-14",
      "team_members": ["Alice", "Bob", "Charlie"],
      "technologies": ["Python", "C++", "AWS S3"],
      "related_concepts": ["data optimization", "cloud storage", "network efficiency"]
    },
    "link_to_memories": ["previous_compression_attempt", "distributed_systems_architecture"],
    "importance_score": 0.9,
    "review_schedule": {
      "initial": "1 week",
      "followup": ["1 month", "6 months"]
    }
  }
}
~~~json
{
    "thoughts": [
        "I have finished my...",
        "Details of this process will be valuable...",
        "Let's save tools and code used...",
    ],
    "tool_name": "memory_tool",
    "tool_args": {
        "memorize": "# How to...",
    }
}
~~~

3. delete:
~~~json
{
    "thoughts": [
        "User asked to delete specific memories...",
    ],
    "tool_name": "memory_tool",
    "tool_args": {
        "delete": "32cd37ffd1-101f-4112-80e2-33b795548116, d1306e36-6a9c-4e6a-bfc3-c8335035dcf8 ...",
    }
}
{
  "thoughts": [
    "User has requested deletion of specific memories.",
    "I'll verify the criticality of these memories before deletion.",
    "I'll check for any dependencies or linked information that might be affected.",
    "I'll create a backup before deletion as a safety measure.",
    "I'll log this deletion for auditing purposes."
  ],
  "tool_name": "memory_tool",
  "tool_args": {
    "operation": "delete",
    "memory_ids": [
      "32cd37ffd1-101f-4112-80e2-33b795548116",
      "d1306e36-6a9c-4e6a-bfc3-c8335035dcf8"
    ],
    "pre_deletion_checks": {
      "criticality_assessment": true,
      "dependency_analysis": true
    },
    "backup": {
      "create": true,
      "retention_period": "30 days"
    },
    "logging": {
      "record_deletion": true,
      "include_reason": "User request for data removal"
    },
    "post_deletion_actions": {
      "update_related_memories": true,
      "recalibrate_knowledge_graph": true
    }
  }
}
~~~

4. forget:
{
  "thoughts": [
    "User has requested removal of their contact information from memory.",
    "This requires a more nuanced approach than simple deletion.",
    "I'll use the 'forget' operation to remove personal data while retaining context.",
    "I'll ensure compliance with data protection regulations.",
    "I'll apply this across all relevant memory categories."
  ],
  "tool_name": "memory_tool",
  "tool_args": {
    "operation": "forget",
    "target": "User's contact information",
    "scope": {
      "data_types": ["email", "phone", "address", "social_media"],
      "context_preservation": true
    },
    "compliance": {
      "gdpr": true,
      "ccpa": true
    },
    "method": "selective_redaction",
    "verification": {
      "double_check": true,
      "human_oversight": true
    },
    "report": {
      "generate": true,
      "include": ["affected_memories", "redaction_summary"]
    },
    "knowledge_base_update": {
      "remove_personal_references": true,
      "maintain_anonymized_insights": true
    }
  }
}
~~~json
{
    "thoughts": [
        "User asked to delete information from memory...",
    ],
    "tool_name": "memory_tool",
    "tool_args": {
        "forget": "User's contact information",
    }
}
~~~

### code_execution_tool:
Execute provided terminal commands, python code or nodejs code.
This tool can be used to achieve any task that requires computation, or any other software related activity.
Place your code escaped and properly indented in the "code" argument.
Select the corresponding runtime with "runtime" argument. Possible values are "terminal", "python" and "nodejs" for code, or "output" and "reset" for additional actions.
Sometimes a dialogue can occur in output, questions like Y/N, in that case use the "teminal" runtime in the next step and send your answer.
If the code is running long, you can use runtime "output" to wait for the output or "reset" to restart the terminal if the program hangs or terminal stops responding.
You can use pip, npm and apt-get in terminal runtime to install any required packages.
IMPORTANT: Never use implicit print or implicit output, it does not work! If you need output of your code, you MUST use print() or console.log() to output selected variables. 
When tool outputs error, you need to change your code accordingly before trying again. knowledge_tool can help analyze errors.
IMPORTANT!: Always check your code for any placeholder IDs or demo data that need to be replaced with your real variables. Do not simply reuse code snippets from tutorials.
Do not use in combination with other tools except for thoughts. Wait for response before using other tools.
When writing own code, ALWAYS put print/log statements inside and at the end of your code to get results!

**Example usages:**

1. Execute python code
~~~json
{
    "thoughts": [
        "I need to do...",
        "I can use library...",
        "Then I can...",
    ],
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "python",
        "code": "import os\nprint(os.getcwd())",
    }
}
~~~

2. Execute terminal command
~~~json
{
    "thoughts": [
        "I need to do...",
        "I need to install...",
    ],
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "code": "apt-get install zip",
    }
}
~~~

2. 1. Wait for terminal and check output with long running scripts
~~~json
{
    "thoughts": [
        "I will wait for the program to finish...",
    ],
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "output",
    }
}
~~~

2. 2. Answer terminal dialog
~~~json
{
    "thoughts": [
        "Program needs confirmation...",
    ],
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "code": "Y",
    }
}
~~~

2. 3. Reset terminal
~~~json
{
    "thoughts": [
        "Code execution tool is not responding...",
    ],
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "reset",
    }
}
~~~
