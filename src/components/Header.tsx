import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Menu, X } from 'lucide-react';
import { useState } from 'react';

const Header = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { to: '/', label: 'Home' },
    { to: '/how-it-works', label: 'How It Works' },
    { to: '/live-demo', label: 'Live Demo' },
    { to: '/model', label: 'Model' },
    { to: '/about', label: 'About' },
  ];

  // Function to scroll to top with smooth animation
  const scrollToTop = () => {
    // Only scroll if not already at the top to avoid redundant animations
    if (window.scrollY > 0) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <header className="bg-[#363636] text-white shadow-lg sticky top-0 z-50 border-b border-yellow-500/20">
      <nav className="container mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center space-x-2" onClick={scrollToTop}>
          <span className="text-2xl font-bold bg-gradient-to-r from-yellow-400 to-yellow-600 bg-clip-text text-transparent">
            ECG Classification
          </span>
        </Link>

        {/* Desktop Menu */}
        <ul className="hidden md:flex items-center space-x-8">
          {navItems.map((item) => (
            <li key={item.to} className=''>
              <Link
                to={item.to}
                className="text-sm font-medium text-gray-200 hover:text-yellow-400 hover:bg-yellow-100/50 px-3 py-2 rounded-md transition-all duration-300 relative group"
                onClick={scrollToTop}
              >
                {item.label}
                <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-yellow-400 transition-all duration-300 group-hover:w-full" />
              </Link>
            </li>
          ))}
        </ul>

        {/* Mobile Menu Button */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden text-gray-200 hover:text-yellow-400"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        >
          {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </Button>
      </nav>

      {/* Mobile Menu */}
      <div
        className={cn(
          'md:hidden bg-[#363636] border-t border-yellow-500/20 transition-all duration-300 ease-in-out',
          isMobileMenuOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0 overflow-hidden'
        )}
      >
        <ul className="flex flex-col items-center py-4 space-y-4">
          {navItems.map((item) => (
            <li key={item.to}>
              <Link
                to={item.to}
                className="text-gray-200 hover:text-yellow-400 hover:bg-yellow-500/10 px-4 py-2 rounded-md text-lg font-medium transition-all duration-300"
                onClick={() => {
                  setIsMobileMenuOpen(false);
                  scrollToTop();
                }}
              >
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </header>
  );
};

export default Header;