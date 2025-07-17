import { useState } from "react";

interface InputBarProps {
    currentMessage: string;
    setCurrentMessage: (message: string) => void;
    onSubmit: (e: React.FormEvent) => void;
}

const InputBar = ({ currentMessage, setCurrentMessage, onSubmit }: InputBarProps) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setCurrentMessage(e.target.value);
    };

    return (
        <div className="p-4 bg-gray-900 border-t border-gray-800 shadow-inner">
            <div className="max-w-4xl mx-auto">
                <div className="flex items-center bg-gray-800 rounded-full p-3 shadow-[0_0_10px_rgba(45,212,191,0.3)] border border-gray-700 focus-within:ring-2 focus-within:ring-teal-400 transition-all duration-300">
                    <button
                        type="button"
                        className="p-2 rounded-full text-gray-400 hover:text-teal-400 hover:bg-gray-700 transition-all duration-200"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    </button>
                    <input
                        type="text"
                        placeholder="Type a message"
                        value={currentMessage}
                        onChange={handleChange}
                        className="flex-grow px-4 py-2 bg-transparent focus:outline-none text-gray-200 placeholder-gray-500"
                    />
                    <button
                        type="button"
                        className="p-2 text-gray-400 hover:text-teal-400 hover:bg-gray-700 rounded-full transition-all duration-200"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                        </svg>
                    </button>
                    <button
                        type="submit"
                        onClick={onSubmit}
                        className="bg-gradient-to-r from-teal-500 to-purple-600 hover:from-teal-600 hover:to-purple-700 rounded-full p-3 ml-2 shadow-[0_0_10px_rgba(45,212,191,0.5)] transition-all duration-200 group"
                    >
                        <svg className="w-6 h-6 text-white transform rotate-45 group-hover:scale-110 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
};

export default InputBar;