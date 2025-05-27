import { useEffect, useRef } from 'react';
import { motion, useAnimation, useInView, useScroll, useTransform } from 'framer-motion';
import { FaHeartbeat } from 'react-icons/fa';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const HomePage = () => {
  const heroControls = useAnimation();
  const datasetControls = useAnimation();
  const mlControls = useAnimation();
  const ctaControls = useAnimation();
  const demoControls = useAnimation();

  const heroRef = useRef(null);
  const datasetRef = useRef(null);
  const mlRef = useRef(null);
  const ctaRef = useRef(null);
  const demoRef = useRef(null);

  const isHeroInView = useInView(heroRef, { once: false, margin: '-100px' });
  const isDatasetInView = useInView(datasetRef, { once: false, margin: '-100px' });
  const isMlInView = useInView(mlRef, { once: false, margin: '-100px' });
  const isCtaInView = useInView(ctaRef, { once: false, margin: '-100px' });
  const isDemoInView = useInView(demoRef, { once: false, margin: '-100px' });

  // Scroll-based blur effect
  const scrollRef = useRef(null); // This ref is used to track the scroll position
  const { scrollY } = useScroll({ container: scrollRef });
  const blurValue = useTransform(scrollY, [200, 300], [0, 10]);
  //https://chatgpt.com/c/68202b4f-f5e8-800a-b8a6-c239def517fe
  const controls = useAnimation();
  useEffect(() => {
    return blurValue.onChange((v) => {
      controls.start({ filter: `blur(${v}px)` });
    });
  }, [blurValue, controls]);

  useEffect(() => {
    heroControls.start({ scale: [1, 1.1, 1], transition: { duration: 1, repeat: Infinity } });
  }, [heroControls]);

  useEffect(() => {
    if (isHeroInView) {
      heroControls.start({ opacity: 1, y: 0, transition: { duration: 0.8 } });
    } else {
      heroControls.start({ opacity: 0, y: 20 });
    }
  }, [isHeroInView, heroControls]);

  useEffect(() => {
    if (isDatasetInView) {
      datasetControls.start({ opacity: 1, x: 0, transition: { duration: 0.5 } });
    } else {
      datasetControls.start({ opacity: 0, x: -20 });
    }
  }, [isDatasetInView, datasetControls]);

  useEffect(() => {
    if (isMlInView) {
      mlControls.start({ opacity: 1, y: 0, transition: { duration: 0.5 } });
    } else {
      mlControls.start({ opacity: 0, y: 20 });
    }
  }, [isMlInView, mlControls]);

  useEffect(() => {
    if (isCtaInView) {
      ctaControls.start({ opacity: 1, scale: 1, transition: { duration: 0.5 } });
    } else {
      ctaControls.start({ opacity: 0, scale: 0.9 });
    }
  }, [isCtaInView, ctaControls]);

  useEffect(() => {
    if (isDemoInView) {
      demoControls.start({ opacity: 1, y: 0, transition: { duration: 0.5 } });
    } else {
      demoControls.start({ opacity: 0, y: 20 });
    }
    console.log("isDemo", isDemoInView)
  }, [isDemoInView, demoControls]);

  return (
    <div className="min-h-screen font-playwrite text-gray-100 relative">
      {/* Full-Screen Background Image with Scroll-Based Blur */}
      {/* chatgpt + grok fix */}
      {/* absolute inset-x-0 top-0 bottom-0 bg-cover bg-center z-0 */}
      {/* fixed inset-0 bg-no-repeat bg-center z-0 */}
      <motion.div
        className="fixed inset-0 bg-no-repeat bg-center z-0"
        style={{ backgroundImage: "url('/la_03_37_20 PM.jpg')",backgroundSize: "cover"}}
        animate={controls} 
        // idk this under can't detect on change scroll on top
        // animate={{ filter: `blur(${blurValue.get()}px)` }}
      />

      {/* Spacer to push content below viewport */}
      <div className="h-screen" />

      {/* Content Wrapper */}
      <div className="relative z-10">
        {/* Hero Section */}
        <motion.section
          ref={heroRef}
          animate={heroControls}
          className="bg-gradient-to-r from-yellow-500 to-amber-800 text-white py-20"
        >
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div animate={heroControls}>
              <FaHeartbeat className="mx-auto h-12 w-12 text-yellow-300" />
            </motion.div>
            <h1 className="mt-4 text-4xl font-extrabold sm:text-5xl">ECG Heartbeat Dataset</h1>
            <p className="mt-2 text-xl sm:text-2xl">Advancing Cardiac Health Through Data</p>
            <p className="mt-4 text-lg max-w-3xl mx-auto">
              Explore a comprehensive collection of heartbeat signals curated for machine learning applications in cardiac analysis, sourced from the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database.
            </p>
          </div>
        </motion.section>

        {/* ECG Explanation */}
        <section className="py-16 bg-[#424242]/80 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-gray-100">What is an ECG?</h2>
            <div className="mt-6 flex flex-col md:flex-row items-center gap-8">
              <div className="md:w-1/2">
                <p className="text-lg text-gray-300">
                  An Electrocardiogram (ECG) measures the heart's electrical activity, helping detect conditions like arrhythmias and myocardial infarction. This dataset segments ECG signals into individual heartbeats, enabling precise classification and analysis for improved cardiac diagnostics.
                </p>
              </div>
              <div className="md:w-1/2">
                <img src="/2858693.svg" alt="ECG Waveform" className="w-full h-auto filter invert" />
              </div>
            </div>
          </div>
        </section>

        {/* Dataset Details */}
        <section ref={datasetRef} className="py-16 bg-[#363636]/80 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-gray-100 text-center">Dataset Components</h2>
            <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-8">
              <motion.div
                animate={datasetControls}
                className="bg-[#424242] p-6 rounded-lg shadow-lg border border-gray-700"
              >
                <h3 className="text-2xl font-semibold text-gray-100">Arrhythmia Dataset</h3>
                <ul className="mt-4 space-y-2 text-gray-300">
                  <li><strong>Samples:</strong> 109,446</li>
                  <li><strong>Categories:</strong> 5</li>
                  <li><strong>Frequency:</strong> 125Hz</li>
                  <li><strong>Source:</strong> MIT-BIH Arrhythmia Dataset</li>
                  <li><strong>Classes:</strong></li>
                  <ul className="ml-4 space-y-1 list-disc">
                    <li>N: Normal beat</li>
                    <li>S: Supraventricular premature beat</li>
                    <li>V: Premature ventricular contraction</li>
                    <li>F: Fusion of ventricular and normal beat</li>
                    <li>Q: Unclassifiable beat</li>
                  </ul>
                </ul>
                <p className="mt-4 text-sm text-gray-400">Preprocessed to 188 dimensions.</p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={datasetControls}
                className="bg-[#424242] p-6 rounded-lg shadow-lg border border-gray-700"
              >
                <h3 className="text-2xl font-semibold text-gray-100">PTB Diagnostic ECG Database</h3>
                <ul className="mt-4 space-y-2 text-gray-300">
                  <li><strong>Samples:</strong> 14,552</li>
                  <li><strong>Categories:</strong> 2</li>
                  <li><strong>Frequency:</strong> 125Hz</li>
                  <li><strong>Source:</strong> PTB Diagnostic Database</li>
                  <li><strong>Classes:</strong> Normal, Abnormal</li>
                </ul>
                <p className="mt-4 text-sm text-gray-400">Preprocessed to 188 dimensions.</p>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Machine Learning Applications */}
        <motion.section
          ref={mlRef}
          animate={mlControls}
          className="py-16 bg-[#424242]/80 backdrop-blur-sm"
        >
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 className="text-3xl font-bold text-gray-100">Machine Learning Applications</h2>
            <p className="mt-6 text-lg text-gray-300 max-w-3xl mx-auto">
              This dataset supports training deep neural networks for heartbeat classification, leveraging its large sample size. It’s been pivotal in transfer learning research, as demonstrated by Kachuee et al. (2018), advancing cardiac diagnostics.
            </p>
            <Button
              asChild
              className="mt-6 bg-[#444444] hover:bg-[#292929] text-white px-8 py-3 rounded-md font-semibold transition"
            >
              <a
                href="https://arxiv.org/abs/1805.00794"
                target="_blank"
                rel="noopener noreferrer"
                className="mt-4 inline-block text-blue-400 hover:underline"
              >
                Read the Research Paper
              </a>
            </Button>
          </div>
        </motion.section>

        {/* Call to Action */}
        <motion.section
          ref={ctaRef}
          animate={ctaControls}
          className="py-16 bg-[#363636]/80 text-white"
        >
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 className="text-3xl font-bold">Ready to Explore?</h2>
            <p className="mt-4 text-lg">Dive into the dataset and build your own ECG classification models.</p>
            <Button
              asChild
              className="mt-6 bg-[#424242] hover:bg-[#232323] text-white px-8 py-3 rounded-md font-semibold transition"
            >
              <a href="#">Get Started</a>
            </Button>
          </div>
        </motion.section>

        {/* Live Demo Section */}
        <motion.section
          ref={demoRef}
          animate={demoControls}
          className="py-16 bg-[#424242]/80 text-white"
        >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold">Live demo</h2>
          <p className="mt-4 text-lg">Time to test it out :D</p>
          <Button
            asChild
            className="mt-6 bg-[#424242] hover:bg-[#232323] text-white px-8 py-3 rounded-md font-semibold transition"
          >
            <Link to="/live-demo">Demo</Link>
          </Button>
        </div>
      </motion.section>

        {/* Footer */}
        <footer className="bg-[#363636]/80 backdrop-blur-sm text-gray-300 py-8">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <p>© 2023 ECG Heartbeat Dataset. All rights reserved.</p>
            <p className="mt-2 text-sm">Sourced from Physionet’s MIT-BIH and PTB Databases.</p>
          </div>
        </footer>
      </div>

      <div className="h-screen" />
    </div>
  );
};

export default HomePage;