// import the modules needed
import express from 'express';
import cors from 'cors';
import fs from 'fs'
import ollama from 'ollama';
import path from 'path';
import { fileURLToPath } from 'url';

// create the application and port to listen to
const app = express();
const PORT = 3000;
const HOST = '127.0.0.1';

// reading the JSON resume file and resume data mapping
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const resumePath = path.join(__dirname, 'info', 'resume.json');
const resume = JSON.parse(fs.readFileSync(resumePath, 'utf-8'));
const resumeSections = {
    summary: `NAME: ${resume.name}\nSUMMARY: ${resume.summary}`,
    education: resume.education.map(e =>
        `DEGREE: ${e.degree} from ${e.school} (${e.completed}) in ${e.location}`).join('\n'),
    skills: `LANGUAGES: ${resume.technicalSkills.languages.join(', ')}\n` +
        `FRAMEWORKS: ${resume.technicalSkills.frameworks.join(', ')}\n` +
        `AI: ${resume.technicalSkills.ai.join(', ')}`,
    projects: resume.projects.map(p =>
        `PROJECT: ${p.project_name}. DESC: ${p.description}. TECH: ${p.technologies_used.join(', ')}`
    ).join('\n\n'),
    certs: resume.certifications.map(c =>
        `CERT: ${c.certification_name} (${c.issue_date})`).join('\n')
};

// regex function
function getRelevantContext(message) {
    const input = message.toLowerCase();
    let knowledgeBase = [];

    const rules = [
        {pattern: /summary|who|about|background/i, content: resumeSections.summary},
        {pattern: /school|college|uni|degree|education|learn/i, content: resumeSections.education},
        {pattern: /skill|tech|stack|code|python|js|javascript|node|express|ai|ml/i, content: resumeSections.skills},
        {pattern: /project|built|make|github|portfolio/i, content: resumeSections.projects},
        {pattern: /cert|aws|linux|license/i, content: resumeSections.certs}
    ];
    rules.forEach(rule => {
        if (rule.pattern.test(input)) {
            knowledgeBase.push(rule.content);
        }
    });

    // middle ground RAG
    // if specific match, return them. if NOT, return general info about me
    if (knowledgeBase.length > 0) {
        return knowledgeBase.join('\n');
    } else {
        return "Sorry, I am still learning. Please ask another question.";
    }
}

const systemPrompt = `
Your name is BadBot. You are a versatile personal assistant.
rules:
1. If the context contains specific data, use it to answer accurately.
2. If the user is just chatting or asking general knowledge, answer using your own internal knowledge.
3. Keep your answers between one to three sentences.
4. Don't mention "Based on the context provided"- just answer naturally.
5. End conversations respectfully.
`


// Cross-Origin Resource Sharing: tell browser it's okay for website to talk to backend
app.use(cors());

app.use(express.json());

app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) {
            return res.status(400).json({error: "Message required"});
        }
        console.log("User: ", message);

        // get the context
        const context = await getRelevantContext(message);

        const response = await ollama.chat({
            model: 'gemma3:1b',
            messages: [
                {
                    role: 'system', content: systemPrompt + context
                },
                {
                    role: 'user', content: message
                }
            ],
            keep_alive: -1 // keeping the model in RAM to see if it helps response timing
        });

        const reply = response?.message?.content || "Sorry, I am still learning. Please ask another question.";
        return res.json({reply});
    } catch (err) {
        console.error("Backend Error:", err);
        return res.status(500).json({error: "Internal Error"});
    }
});

app.listen(PORT, () => {
    console.log(`Server running internally at http://${HOST}:${PORT}`);
});