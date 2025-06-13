import { useState } from 'react';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className="bg-gray-950/90 backdrop-blur-sm text-white shadow-[0_0_15px_rgba(45,212,191,0.4)] border-b border-teal-500/30">
      <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <svg className="w-10 h-10 animate-pulse text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h1 className="text-3xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-purple-500 font-poppins">
            Vasavi GenZ
          </h1>
        </div>
        <nav className="hidden md:flex space-x-8">
          {['Home', 'Chat', 'Trends', 'Profile'].map((item) => (
            <a
              key={item}
              href="#"
              className="text-sm font-medium text-gray-200 hover:text-teal-300 px-4 py-2 rounded-xl transition-all duration-300 transform hover:scale-105 hover:bg-gray-800/50 shadow-[0_0_8px_rgba(45,212,191,0.3)]"
            >
              {item.toUpperCase()}
            </a>
          ))}
        </nav>
        <button
          className="md:hidden p-3 rounded-full text-gray-200 hover:text-teal-300 hover:bg-gray-800/50 transition-all duration-300"
          onClick={toggleMenu}
        >
          <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
      {isMenuOpen && (
        <div className="md:hidden bg-gray-900/95 backdrop-blur-sm px-6 py-4 flex flex-col space-y-4">
          {['Home', 'Chat', 'Trends', 'Profile'].map((item) => (
            <a
              key={item}
              href="#"
              className="text-sm font-medium text-gray-200 hover:text-teal-300 px-4 py-2 rounded-xl transition-all duration-300 transform hover:scale-105 hover:bg-gray-800/50"
              onClick={toggleMenu}
            >
              {item.toUpperCase()}
            </a>
          ))}
        </div>
      )}
    </header>
  );
};

export default Header;