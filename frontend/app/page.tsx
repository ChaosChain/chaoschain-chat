"use client";
import { Thread } from "@/components/assistant-ui/thread";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { AssistantRuntimeProvider } from "@assistant-ui/react";

export default function Home() {
  const runtime = useChatRuntime({ api: "http://localhost:8000/api/chat" });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex flex-col h-dvh bg-gray-900 text-gray-200">
        <header className="p-3 bg-gray-900 border-b border-gray-800 text-center flex-shrink-0">
           <h1 className="text-lg font-semibold">Chaos Chain Lite Paper Chat</h1>
        </header>

        <main className="flex-grow px-4 py-4 overflow-hidden flex flex-col">
          <div className="bg-gray-900 overflow-y-auto flex-grow">
             <Thread />
           </div>
        </main>
       </div>
    </AssistantRuntimeProvider>
  );
}
