import { FaGithub, FaFacebook, FaXingSquare } from 'react-icons/fa';
import { motion } from 'framer-motion';

const Footer = () => {
  // Animation variants for staggered fade-in
  const containerVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        staggerChildren: 0.15,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 5 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  };

  return (
    <motion.nav
      className="fixed bottom-0 left-0 w-full bg-gradient-to-r from-gray-950 to-gray-900 text-white h-16 flex items-center justify-center shadow-lg z-50"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      <div className="container mx-auto px-4 flex items-center justify-between">
        {/* Project Info */}
        <motion.div variants={itemVariants} className="flex items-center space-x-2">
          <h3 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-yellow-500">
            ECG Heartbeat
          </h3>
          <p className="text-amber-200 text-sm hidden md:block">
            Machine Learning-driven ECG Analysis
          </p>
        </motion.div>

        {/* Social Media & Contact */}
        <motion.div variants={itemVariants} className="flex items-center space-x-4">
          <div className="flex space-x-3">
            {[
              { Icon: FaXingSquare, href: 'https://x.com/me_siuuuu', label: 'Twitter' },
              { Icon: FaGithub, href: 'https://github.com/trungdangtapcode', label: 'GitHub' },
              { Icon: FaFacebook, href: 'https://www.facebook.com/ToRungLa', label: 'Facebook' },
            ].map(({ Icon, href, label }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-amber-200 hover:text-yellow-400 transition-colors duration-200 transform hover:scale-110"
                aria-label={label}
              >
                <Icon size={20} />
              </a>
            ))}
          </div>
          <a
            href="mailto:23521684@gm.uit.edu.vn"
            className="text-amber-200 text-sm hover:text-yellow-400 transition-colors hidden md:block"
          >
            23521684@gm.uit.edu.vn
          </a>
          <a
            href="mailto:23521719@gm.uit.edu.vn"
            className="text-amber-200 text-sm hover:text-yellow-400 transition-colors hidden md:block"
          >
            23521719@gm.uit.edu.vn
          </a>
          <a
            href="mailto:23521689@gm.uit.edu.vn"
            className="text-amber-200 text-sm hover:text-yellow-400 transition-colors hidden md:block"
          >
            23521689@gm.uit.edu.vn
          </a>
          <a
            href="mailto:23521714@gm.uit.edu.vn"
            className="text-amber-200 text-sm hover:text-yellow-400 transition-colors hidden md:block"
          >
            23521714@gm.uit.edu.vn
          </a>
          
        </motion.div>

        {/* Copyright & Tech */}
        <motion.div variants={itemVariants} className="text-amber-300 text-sm hidden lg:block">
          <p>
            © {new Date().getFullYear()} | Built with{' '}
            <a
              href="https://vitejs.dev"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-yellow-400 transition-colors"
            >
              Vite
            </a>{' '}
            &{' '}
            <a
              href="https://ui.shadcn.com"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-yellow-400 transition-colors"
            >
              Shadcn
            </a>
          </p>
        </motion.div>
      </div>
    </motion.nav>
  );
};

export default Footer;