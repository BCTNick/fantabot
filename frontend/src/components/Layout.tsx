import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-primary text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">⚽ FantaBot</h1>
          <p className="text-gray-300 mt-1">Fantasy Football Auction Manager</p>
        </div>
      </header>
      
      <main className="container mx-auto px-4 py-6">
        {children}
      </main>
      
      <footer className="bg-secondary text-white mt-12">
        <div className="container mx-auto px-4 py-4 text-center">
          <p className="text-gray-400">&copy; 2025 FantaBot - Fantasy Football Auction System</p>
        </div>
      </footer>
    </div>
  );
};
