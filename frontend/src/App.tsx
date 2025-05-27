
import './App.css'

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import InteractivePage from './InteractivePage/page';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import HowItWorksPage from './pages/HowItWorksPage';
// import LiveDemoPage from './pages/LiveDemoPage';
import AboutPage from './pages/AboutPage';
import CodeShowcase from './pages/ShowCase';
function App() {

  return (
    
    <Router>
    {/* manually min-w-300, I don't know how to fix :( */}
    <div className="min-h-screen flex flex-col min-w-300">
    <Header/>
    {/* flex-grow idk how I work :D */}
    <main className="flex-grow">
      {/* ---------------------------------------------------------------------------------------------------------------------------------------------- */}
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/how-it-works" element={<HowItWorksPage />} />
        {/* <Route path="/live-demo" element={<LiveDemoPage />} /> */}
        <Route path="/model" element={<CodeShowcase />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/live-demo" element={<InteractivePage />} />
      </Routes>
    
    </main>
    <Footer />
  </div>
  </Router>
  )
}

export default App
