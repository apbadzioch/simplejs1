import ollama from 'ollama'
import readline from 'readline'
import fs from 'fs'

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const resume = JSON.parse(fs.readFileSync('./resume.json'));

const resumeContext = `
name: ${resume.name}
email: ${resume.email}
education: ${resume.education.map(e => e.degree).join(', ')}
tech: ${resume.technical_foundations}
person: ${resume.personal_foundations}
work: ${resume.work_experience}
projects: ${resume.projects}
certs: ${resume.certifications}
`

const systemPrompt = `
You are a personal assistant for Andrew.
Be concise with you answers: 1-2 sentences MAX.
Don not add extra content or suggestions that do not help answer the question.`

// as of 11-18-2025 this version of gemma3:1b has been trained up till November 2, 2023
// this was found out when I asked about the 2025 world series results

// similar to "while true:" in Python
async function ask() {
    rl.question('> ', async(userInput) => {
        if (userInput.toLowerCase() === 'quit') {
            rl.close()
            return
        }

        const response = await ollama.chat({
            model: 'gemma3:1b',
            messages: [
                {
                    role: 'system', content: systemPrompt + resumeContext
                },
                {
                    role: 'user', content: userInput
                }
            ]
        })

        console.log(response.message.content)
        ask() // loop again
    })
}
ask()
/*
synchronous programming: task are executed sequentially, with each operation
waiting for the previous one to complete before proceeding, ensuring predictability
and simplicity in code execution, but may lead to inefficiencies when dealing with
time-consuming operations.

asynchronous programming: allows tasks to run concurrently, enabling non-blocking
execution and better resource utilization. By using callbacks, promises, and
async/await syntax, asynchronous programming enhances app responsiveness and scalability.
*/