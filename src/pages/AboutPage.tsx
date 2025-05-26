import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const AboutPage = () => {
  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8, ease: "easeOut" },
    },
  };

  const imageVariants = {
    hover: { scale: 1.05, rotate: 5, transition: { duration: 0.3 } },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-[#363636] to-gray-800 flex items-center justify-center p-4 sm:p-8">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-5xl w-full"
      >
        <Card className="bg-black/80 backdrop-blur-md border border-yellow-500/20 shadow-xl hover:shadow-yellow-500/10 transition-shadow duration-500">
          <CardContent className="p-6 sm:p-12 flex flex-col md:flex-row items-center gap-8">
            <motion.div
              variants={imageVariants}
              whileHover="hover"
              className="relative"
            >
              <img
                src="/wtm_about.png"
                alt="About the ECG heartbeat classifier"
                className="w-56 h-56 object-cover rounded-full border-4 border-yellow-400 shadow-lg"
              />
              <div className="absolute inset-0 rounded-full bg-yellow-400/20 blur-xl animate-pulse" />
            </motion.div>

            <div className="text-yellow-300 space-y-6">
              <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-yellow-300 to-yellow-500">
                About This Project
              </h1>

              <p className="text-lg sm:text-xl leading-relaxed text-yellow-200/90">
                Crafted by{" "}
                <span className="font-semibold text-yellow-400">Me</span>. This
                application showcases ECG heartbeat classification using advanced
                machine-learning techniques, covering everything from data
                preprocessing and feature extraction to precise rhythm
                classification.
              </p>

              <Button
                asChild
                className="bg-yellow-400 hover:bg-yellow-300 text-black font-semibold py-3 px-6 rounded-lg shadow-md hover:shadow-yellow-400/50 transition-all duration-300"
              >
                <a
                  href="https://github.com/trungdangtapcode/ecg-heartbeat-classifier"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2"
                >
                  <span>🔗 View GitHub Repository</span>
                </a>
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default AboutPage;