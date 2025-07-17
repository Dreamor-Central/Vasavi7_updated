import React from 'react';

interface SearchInfo {
    stages: string[];
    query?: string;
    urls?: string[] | string;
    error?: string | object;
}

interface Message {
    id: number;
    content: string | object;
    isUser: boolean;
    type: string;
    isLoading?: boolean;
    agent?: string;
    searchInfo?: SearchInfo;
}

interface SearchStagesProps {
    searchInfo: SearchInfo;
}

interface MessageAreaProps {
    messages: Message[];
}

const PremiumTypingAnimation = () => {
    return (
        <div className="flex items-center space-x-2">
            <div className="flex space-x-1.5">
                {[0, 300, 600].map((delay, index) => (
                    <div
                        key={index}
                        className="w-2 h-2 bg-teal-400 rounded-full animate-bounce shadow-[0_0_8px_rgba(45,212,191,0.6)]"
                        style={{ animationDuration: '0.8s', animationDelay: `${delay}ms` }}
                    ></div>
                ))}
            </div>
            <span className="text-sm text-gray-400 italic">Typing...</span>
        </div>
    );
};

const SearchStages = ({ searchInfo }: SearchStagesProps) => {
    if (!searchInfo || !searchInfo.stages || searchInfo.stages.length === 0) return null;

    return (
        <div className="mb-4 mt-2 relative pl-6">
            <div className="flex flex-col space-y-5 text-sm text-gray-400">
                {searchInfo.stages.includes('searching') && (
                    <div className="relative">
                        <div className="absolute -left-4 top-2 w-3 h-3 bg-teal-400 rounded-full shadow-[0_0_10px_rgba(45,212,191,0.8)] animate-pulse"></div>
                        {searchInfo.stages.includes('reading') && (
                            <div className="absolute -left-[15px] top-4 w-1 h-[calc(100%+1.5rem)] bg-gradient-to-b from-teal-400/50 to-purple-500/50"></div>
                        )}
                        <div className="flex flex-col">
                            <span className="font-semibold text-teal-400 ml-3">Searching the web</span>
                            <div className="flex flex-wrap gap-3 pl-3 mt-2">
                                <div className="bg-gray-800 text-xs px-4 py-2 rounded-lg border border-gray-700 flex items-center shadow-[0_0_5px_rgba(45,212,191,0.3)] hover:bg-gray-700 transition-all duration-200">
                                    <svg className="w-4 h-4 mr-2 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                    </svg>
                                    {searchInfo.query}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
                {searchInfo.stages.includes('reading') && (
                    <div className="relative">
                        <div className="absolute -left-4 top-2 w-3 h-3 bg-teal-400 rounded-full shadow-[0_0_10px_rgba(45,212,191,0.8)] animate-pulse"></div>
                        <div className="flex flex-col">
                            <span className="font-semibold text-teal-400 ml-3">Reading results</span>
                            {searchInfo.urls && searchInfo.urls.length > 0 && (
                                <div className="pl-3 mt-2 space-y-2">
                                    <div className="flex flex-wrap gap-3">
                                        {Array.isArray(searchInfo.urls) ? (
                                            searchInfo.urls.map((url, index) => (
                                                <a
                                                    key={index}
                                                    href={typeof url === 'string' ? url : '#'}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="bg-gray-800 text-xs px-4 py-2 rounded-lg border border-gray-700 truncate max-w-[250px] hover:bg-gray-700 transition-all duration-200 shadow-[0_0_5px_rgba(45,212,191,0.3)] text-gray-300"
                                                >
                                                    {typeof url === 'string' ? url : JSON.stringify(url).substring(0, 40)}
                                                </a>
                                            ))
                                        ) : (
                                            <div className="bg-gray-800 text-xs px-4 py-2 rounded-lg border border-gray-700 truncate max-w-[250px] hover:bg-gray-700 transition-all duration-200 shadow-[0_0_5px_rgba(45,212,191,0.3)] text-gray-300">
                                                {searchInfo.urls 
                                                    ? (typeof searchInfo.urls === 'string' 
                                                        ? searchInfo.urls.substring(0, 40) 
                                                        : JSON.stringify(searchInfo.urls).substring(0, 40))
                                                    : "No URLs"}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
                {searchInfo.stages.includes('writing') && (
                    <div className="relative">
                        <div className="absolute -left-4 top-2 w-3 h-3 bg-teal-400 rounded-full shadow-[0_0_10px_rgba(45,212,191,0.8)] animate-pulse"></div>
                        <span className="font-semibold text-teal-400 ml-3">Writing answer</span>
                    </div>
                )}
                {searchInfo.stages.includes('error') && (
                    <div className="relative">
                        <div className="absolute -left-4 top-2 w-3 h-3 bg-red-500 rounded-full shadow-[0_0_10px_rgba(239,68,68,0.8)]"></div>
                        <span className="font-semibold text-red-400 ml-3">Search error</span>
                        <div className="pl-5 text-xs text-red-400 mt-1.5">
                            {searchInfo.error 
                                ? (typeof searchInfo.error === 'string' 
                                    ? searchInfo.error 
                                    : JSON.stringify(searchInfo.error)) 
                                : "An error occurred during search."}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

const MessageArea = ({ messages }: MessageAreaProps) => {
    return (
        <div className="flex-grow overflow-y-auto bg-gradient-to-b from-gray-900 to-gray-800" style={{ minHeight: 0 }}>
            <div className="max-w-4xl mx-auto p-6 space-y-6">
                {messages.map((message) => (
                    <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-6`}>
                        <div className="flex flex-col max-w-lg">
                            {!message.isUser && message.searchInfo && (
                                <SearchStages searchInfo={message.searchInfo} />
                            )}
                            <div
                                className={`rounded-2xl p-4 ${message.isUser
                                    ? 'bg-gradient-to-br from-teal-500 to-purple-600 text-white shadow-[0_0_10px_rgba(45,212,191,0.5)]'
                                    : 'bg-gray-800 text-gray-200 border border-gray-700 shadow-[0_0_5px_rgba(0,0,0,0.3)]'
                                } transition-all duration-200 hover:shadow-[0_0_15px_rgba(45,212,191,0.7)]`}
                            >
                                {message.isLoading ? (
                                    <PremiumTypingAnimation />
                                ) : (
                                    <p className="text-sm leading-relaxed">
                                        {message.content ? (
                                            typeof message.content === 'string' 
                                                ? message.content 
                                                : JSON.stringify(message.content)
                                        ) : (
                                            <span className="text-gray-500 text-xs italic">Waiting for response...</span>
                                        )}
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MessageArea;