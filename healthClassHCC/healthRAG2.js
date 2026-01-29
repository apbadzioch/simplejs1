import path from 'path';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OllamaEmbeddings, Ollama } from '@langchain/ollama';
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';
import { RetrievalQAChain } from '@langchain/classic/chains';
import { Chroma } from "@langchain/community/vectorstores/chroma";

async function buildLangChainRAG(pdfFiles) {
    console.log("Loading...");
    let allDocs = [];

    for (const file of pdfFiles) {
        const resolvedPath = path.resolve(file);
        const loader = new PDFLoader(resolvedPath);
        const docs = await loader.load();
        allDocs.push(...docs);
    }

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 50,
    });

    const splits = await splitter.splitDocuments(allDocs);

    const embeddings = new OllamaEmbeddings({
        model: "nomic-embed-text",
        baseUrl: "http://127.0.0.1:11434"
    });

    const llm = new Ollama({
        model: "gemma3:1b",
        baseUrl: "http://127.0.0.1:11434",
        temperature: 0.2
    })

    const vectorstore = await MemoryVectorStore.fromDocuments(splits, embeddings);

    const chain = RetrievalQAChain.fromLLM(llm, vectorstore.asRetriever(3));
    return chain;
};

async function main() {
    try {
        const diabetesChain = await buildLangChainRAG([
            "data/diabetes.pdf",
            "data/standards.pdf"
        ]);

        const queries = [
            "What are metformin side effects?",
            "A1C target range for type 2 diabetes?",
            "How to treat hypoglycemia?",
            "When to check blood glucose?",
            "Foot care recommendations for diabetics?"
        ];

        for (const q of queries) {
            console.log(`\nQuestion: ${q}`);
            const start = performance.now();
            const res = await diabetesChain.call({ query: q });
            const duration = performance.now() - start;

            console.log("LangChain:", res.text);
            console.log("Time:", duration);
        }
    } catch (error) {
        console.error(error);
    }
}

main()