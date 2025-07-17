// src/app/page.tsx
"use client"

import Header from '@/components/Header';
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState } from 'react';

interface Message {
  id: number;
  content: string | object; // Support string or JSON object for agent responses
  isUser: boolean;
  type: string;
  isLoading?: boolean;
  agent?: string; // Track which agent/tool responded (sales, recommendation, styling, trend)
  searchInfo?: { // Add this interface for search stages
    stages: string[];
    query?: string;
    urls?: string[];
    error?: string | object;
  };
}

const Home = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: 'Hi there, how can I help you? Ask about products, styling, trends, or sales!',
      isUser: false,
      type: 'message'
    }
  ]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [checkpointId, setCheckpointId] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (currentMessage.trim()) {
      // Add user message to the chat
      const newMessageId = messages.length > 0 ? Math.max(...messages.map(msg => msg.id)) + 1 : 1;

      setMessages(prev => [
        ...prev,
        {
          id: newMessageId,
          content: currentMessage,
          isUser: true,
          type: 'message'
        }
      ]);

      const userInput = currentMessage;
      setCurrentMessage(""); // Clear input field

      try {
        // Create AI response placeholder
        const aiResponseId = newMessageId + 1;
        setMessages(prev => [
          ...prev,
          {
            id: aiResponseId,
            content: "",
            isUser: false,
            type: 'message',
            isLoading: true, // Set isLoading to true initially for the AI response
            agent: undefined, // Ensure agent is undefined initially
            searchInfo: { stages: [] } // Initialize searchInfo to prevent errors
          }
        ]);

        // Use confirmed EC2 endpoint
        let url = `http://localhost:8001/chat/stream?message=${encodeURIComponent(userInput)}`;
        if (checkpointId) {
          url += `&checkpoint_id=${encodeURIComponent(checkpointId)}`;
        }

        const response = await fetch(url, {
          method: 'POST',
        });

        if (!response.ok) {
          const errorDetail = await response.text();
          throw new Error(`HTTP error! Status: ${response.status}. Details: ${errorDetail || 'No additional error details.'}`);
        }

        let streamedContent: string | object = "";
        let currentAgent: string | undefined = undefined;
        // let hasReceivedContent = false; // Not strictly needed with the current update logic
        let currentSearchInfo = { stages: [] as string[], query: '', urls: [] as string[], error: '' }; // Initialize for streaming updates

        const contentType = response.headers.get('content-type');

        if (contentType && contentType.includes('text/event-stream')) {
          const reader = response.body?.getReader();
          if (reader) {
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });

              const events = buffer.split('\n\n');
              buffer = events.pop() || '';

              for (const eventString of events) {
                if (eventString.trim() === '') continue;

                const lines = eventString.split('\n');
                let eventData: { type?: string; content?: string | object; [key: string]: any; } = {};

                for (const line of lines) {
                  if (line.startsWith('data:')) {
                    try {
                      // Attempt to parse data as JSON
                      eventData = JSON.parse(line.substring(5).trim());
                    } catch (parseError) {
                      // If parsing fails, treat content as a plain string
                      console.error("Error parsing event data line:", parseError, line);
                      eventData.type = 'content';
                      eventData.content = line.substring(5).trim();
                    }
                  }
                }

                if (eventData.type === 'checkpoint') {
                  setCheckpointId(eventData.content as string);
                } else if (eventData.type === 'success' || eventData.type === 'content') {
                  let newContent: string | object = typeof eventData.content === 'object' ? eventData.content : String(eventData.content);

                  if (typeof newContent === 'string' && typeof streamedContent === 'string') {
                    streamedContent += newContent;
                  } else {
                    // If content type changes (e.g., from text to JSON object), replace
                    streamedContent = newContent;
                  }
                  // hasReceivedContent = true; // No longer strictly needed

                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === aiResponseId
                        ? { ...msg, content: streamedContent, isLoading: false, agent: currentAgent, searchInfo: currentSearchInfo }
                        : msg
                    )
                  );
                } else if (eventData.type === 'tool_code') {
                    // Handle tool_code events, especially for search stages
                    const toolCallData = typeof eventData.content === 'object' ? eventData.content : JSON.parse(eventData.content as string);

                    if (toolCallData.tool_name === 'search') {
                        if (toolCallData.stage === 'searching') {
                            currentSearchInfo = { ...currentSearchInfo, stages: ['searching'], query: toolCallData.query };
                        } else if (toolCallData.stage === 'reading') {
                            currentSearchInfo = { ...currentSearchInfo, stages: ['searching', 'reading'], urls: toolCallData.urls };
                        } else if (toolCallData.stage === 'writing') {
                            currentSearchInfo = { ...currentSearchInfo, stages: ['searching', 'reading', 'writing'] };
                        } else if (toolCallData.stage === 'error') {
                            currentSearchInfo = { ...currentSearchInfo, stages: ['error'], error: toolCallData.error };
                        }
                    }
                    // Update the message with the current search info, and keep loading true
                    setMessages(prev =>
                        prev.map(msg =>
                            msg.id === aiResponseId
                                ? { ...msg, agent: currentAgent, isLoading: true, searchInfo: { ...currentSearchInfo } }
                                : msg
                        )
                    );
                } else if (eventData.type === 'tool_call') {
                  // This branch handles when the agent name changes, separate from search stages
                  currentAgent = eventData.content as string;
                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === aiResponseId
                        ? { ...msg, agent: currentAgent, isLoading: true, searchInfo: currentSearchInfo }
                        : msg
                    )
                  );
                } else if (eventData.type === 'error') {
                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === aiResponseId
                        ? { ...msg, content: eventData.content || "An error occurred during streaming.", isLoading: false, agent: currentAgent, searchInfo: { ...currentSearchInfo, stages: ['error'], error: typeof eventData.content === 'string' ? eventData.content : JSON.stringify(eventData.content) || "Unknown error" } }
                        : msg
                    )
                  );
                  reader.cancel();
                } else if (eventData.type === 'end') {
                  setMessages(prev =>
                    prev.map(msg =>
                      msg.id === aiResponseId
                        ? { ...msg, isLoading: false, agent: currentAgent, searchInfo: currentSearchInfo }
                        : msg
                    )
                  );
                  reader.cancel();
                  break;
                }
              }
            }
          }
        } else {
          // If the response is not event-stream, just read it as JSON or text
          const finalResponseData = await response.json(); // Adjust to .text() if not JSON
          streamedContent = finalResponseData.content || "No content received.";

          setMessages(prev =>
            prev.map(msg =>
              msg.id === aiResponseId
                ? { ...msg, content: streamedContent, isLoading: false, agent: finalResponseData.agent || currentAgent, searchInfo: currentSearchInfo }
                : msg
            )
          );
        }

      } catch (error) {
        console.error("Error during chat request:", error);
        setMessages(prev => [
          ...prev,
          {
            id: prev.length > 0 ? Math.max(...prev.map(msg => msg.id)) + 1 : 1,
            content: `Sorry, there was an error processing your request: ${error instanceof Error ? error.message : String(error)}`,
            isUser: false,
            type: 'message',
            isLoading: false,
            searchInfo: { stages: ['error'], error: error instanceof Error ? error.message : String(error) }
          }
        ]);
      }
    }
  };

  return (
    <div className="flex justify-center bg-gray-950 min-h-screen py-8 px-4 text-gray-100">
      <div className="w-full max-w-4xl bg-gray-900 flex flex-col rounded-xl shadow-2xl border border-gray-700 overflow-hidden h-[90vh]">
        <Header />
        <MessageArea messages={messages} />
        <InputBar currentMessage={currentMessage} setCurrentMessage={setCurrentMessage} onSubmit={handleSubmit} />
      </div>
    </div>
  );
};

export default Home;