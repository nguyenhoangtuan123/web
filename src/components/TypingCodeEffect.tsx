import { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface TypingEffectProps {
  title: string;
  code: string;
}

const useTypingEffect = (text: string, speed: number = 5): string => {
  const [displayedText, setDisplayedText] = useState<string>('');
  const [index, setIndex] = useState<number>(0);
  const [isTyping, setIsTyping] = useState<boolean>(true);
  const [erasePercent, setErasePercent] = useState<number>(0);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isTyping) {
      if (index < text.length) {
        timer = setTimeout(() => {
          setDisplayedText((prev) => prev + text[index]);
          setIndex(index + 1);
        }, speed);
      } else {
        const randomPercent: number = Math.floor(Math.random() * 60) + 20; // 20% to 80%
        setErasePercent(randomPercent);
        setIsTyping(false);
      }
    } else {
      const charsToErase: number = Math.floor((erasePercent / 100) * text.length);
      if (index > text.length - charsToErase) {
        timer = setTimeout(() => {
          setDisplayedText((prev) => prev.slice(0, -1));
          setIndex(index - 1);
        }, speed / 2);
      } else {
        setIsTyping(true);
      }
    }
    return () => clearTimeout(timer);
  }, [index, isTyping, text, speed, erasePercent]);

  return displayedText;
};

const CodeTypingEffect = ({ title, code }: TypingEffectProps) => {
  const displayedCode = useTypingEffect(code);

  return (
    <div className="p-6 bg-[#1a1a1a] rounded-xl w-full max-w-2xl text-left flex flex-col items-start shadow-lg border border-[#333]">
      <h2 className="text-2xl font-bold text-[#facc15] mb-4 w-full drop-shadow">
        {title}
      </h2>
      <div className="bg-[#0d0d0d] text-[#e5e5e5] font-mono text-sm rounded-lg overflow-x-auto w-full p-4 leading-relaxed">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{
            margin: 0,
            padding: '1rem',
            background: 'transparent',
            fontSize: '0.875rem',
            textAlign: 'left',
            display: 'block',
          }}
          wrapLines={true}
        >
          {displayedCode || ' '}
        </SyntaxHighlighter>
        <span className="inline-block w-[2px] h-[1em] bg-white animate-pulse -ml-[2px] relative -top-[1em]" />
      </div>
    </div>
  );
};

export default CodeTypingEffect;