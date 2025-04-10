"use client";
import { Thread } from "@/components/assistant-ui/thread";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { ThreadList } from "@/components/assistant-ui/thread-list";
// import ChatInterface from '../components/assistant-ui/ChatInterface' // Removed incorrect import

export default function Home() {
  // Use environment variable for API URL, fallback to local default
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/chat";
  const runtime = useChatRuntime({ api: apiUrl });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-screen bg-zinc-950 text-zinc-100">
        <div className="flex-1 flex flex-col">
          {/* <ChatInterface /> */} {/* Removed incorrect component usage */}
          <header className="p-3 border-b text-center flex-shrink-0">
             <h1 className="text-lg font-semibold">Chaos Chain Chat</h1>
          </header>

          <main className="flex-grow px-4 py-4 overflow-hidden flex flex-col">
            <div className="overflow-y-auto flex-grow">
               <Thread />
             </div>
          </main>
        </div>
       </div>
    </AssistantRuntimeProvider>
  );
}
