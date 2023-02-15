import Head from "next/head";

interface SEOProps {
  title?: string | undefined | null;
  description?: string | undefined | null;
  image?: string | undefined | null;
  url?: string | undefined | null;
}

const BASE_URL = "https://hatai-ai.netlify.app";
const fallbacks = {
  title: "หทัย AI: AI ทำนายผลภาพสแกนคลื่นไฟฟ้าหัวใจ",
  description:
    "หทัย AI เป็นแอพพลิเคชั่นที่ช่วยทำนายผลภาพสแกนคลื่นไฟฟ้าหัวใจแบบ 12 ลีด ร่วมกับแบบประเมินความเสี่ยงภาวะโรคหัวใจต่าง ๆ",
  image: "/images/og-image.png",
  url: "https://hatai-ai.netlify.app",
};

const SEO = ({ title, description, image }: SEOProps) => {
  return (
    <Head>
      <title>{title || fallbacks.title}</title>
      <meta name="twitter:card" content="summary" key="twcard" />
      <meta
        property="og:title"
        content={title || fallbacks.title}
        key="ogtitle"
      />
      <meta
        property="og:description"
        content={description || fallbacks.description}
        key="ogdesc"
      />
      <meta
        property="og:image"
        content={`${BASE_URL}${image || fallbacks.image}`}
        key="ogimage"
      />
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="600" />
      <meta
        property="og:site_name"
        content={fallbacks.title}
        key="ogsitename"
      />
    </Head>
  );
};

export default SEO;
