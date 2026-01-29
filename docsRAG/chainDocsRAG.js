/*
ChatOllama from the LangChain DOC pages.
- Make sure all resources are installed
*/
/*
1. ingestion and extraction
2. indexing (embedding and storage)
3. retrieval
4. augmentation and prompt engineering
5. generation
*/

import { ChatOllama } from "@langchain/ollama"; // # 5
import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory"; // will load the entire directory, then follow with what is needed
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"; // # 1
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"; // # 1
import { OllamaEmbeddings } from "@langchain/ollama"; // # 3
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory"; // # 3
// -------------------------------------------------------------

// --- Loading Documents ---
const sugarPdfPath = "AWS_SAA_C03_Cheatsheets.pdf"
const loader = new PDFLoader(sugarPdfPath);

const docs = await loader.load()


// --- Splitting the Documents ---
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 50,
    chunkOverlap: 5
})
const chunks = await splitter.splitDocuments(docs);


// -------------------------------------------------------------

// --- Embedding (Indexing) ---
const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434"
});

// --- Vector Store in Memory ---
const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// -------------------------------------------------------------

// --- Retrieval ---
const query = "what is aws s3 used for?";

const results = await vectorStore.similaritySearch(query, 3);

// -------------------------------------------------------------

// --- Generation ---
const context = results.map(r => r.pageContent).join("\n\n");

const prompt = `
only answer like below.

Query: ${query}
Context: ${context}

`;

// --- Base LLM ---
// instantiate the llm model and generate chat completion.
const llm = new ChatOllama({
    model: "gemma3:1b",
    temperature: 0.3
});

const answer = await llm.invoke(prompt);
console.log(answer);






// console.log(aiMsg.content)
// console.log(docs[0].metadata)