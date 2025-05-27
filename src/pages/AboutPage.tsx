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

            <div className="text-yellow-300 space-y-6">
              <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-yellow-300 to-yellow-500">
                About This Project
              </h1>

              <p className="text-lg sm:text-xl leading-relaxed text-yellow-200/90">
                <span className="font-semibold text-yellow-400"></span>.Chúng tôi là Nhóm 11, gồm các thành viên đang theo học học phần  Machine Learning tại UIT.
                Trong khuôn khổ báo cáo cuối kỳ, nhóm đã cùng nhau nghiên cứu, phát triển và hoàn thiện đề tài: "Phân Loại Nhịp Tim Sử Dụng Các Thuật Toán Máy học 
                và So Sánh Giữa Thư Viện Tối Ưu và Cài Đặt Thủ Công."
              </p>

              <Button
                asChild
                className="bg-yellow-400 hover:bg-yellow-300 text-black font-semibold py-3 px-6 rounded-lg shadow-md hover:shadow-yellow-400/50 transition-all duration-300"
              >
                <a
                  href="https://drive.google.com/file/d/1WuGlTLZwMGODoK1ZKf8vJWcSdhAvwJWZ/view?usp=drive_link"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2"
                >
                  <span>🔗 Xem báo cáo</span>
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