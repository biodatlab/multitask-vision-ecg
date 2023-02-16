import { Box, Flex, Heading, Image, Link, Stack, Text } from "@chakra-ui/react";
import Layout from "../components/layout";

const About = () => {
  return (
    <Layout>
      {/* hacky bg color */}
      <Box>
        <Flex
          bg="secondary.50"
          w="100vw"
          h="100vh"
          position="fixed"
          top={0}
          left={0}
          zIndex={-1}
        />
      </Box>

      {/* main container */}
      <Box my={12}>
        {/* header block */}
        <Box
          bgColor="primary.50"
          bgImage={{
            base: "/images/about-ecg-signal-small.png",
            md: "/images/about-ecg-signal.png",
          }}
          bgRepeat="repeat-x"
          bgPosition="center"
          bgSize="100% 75%"
          borderRadius="xl"
          py={{ base: 8, md: 10 }}
          px={{ base: 6, md: 12, lg: 16 }}
          mb={4}
        >
          <Stack
            direction={{ base: "column", md: "row" }}
            gap={{ base: 2, md: 12, lg: 16 }}
          >
            <Box minW={48} my="auto">
              <Heading
                w="max-content"
                as="h1"
                fontSize={{ base: "4xl", md: 40 }}
                fontWeight={600}
                color="secondary.400"
                lineHeight="tall"
              >
                เกี่ยวกับเรา
              </Heading>

              <Heading
                as="h3"
                fontSize={{ base: "xl", md: "2xl" }}
                color="primary.300"
              >
                หทัย AI
              </Heading>
            </Box>

            <Box
              pt={{ base: 0, md: 2 }}
              sx={{
                marginTop: "auto !important",
                marginBottom: "auto !important",
              }}
            >
              <Heading
                as="h3"
                fontSize={{ base: "xl", md: "2xl" }}
                fontWeight={400}
                color="secondary.400"
                lineHeight="9"
              >
                เพิ่มประสิทธิภาพการอ่านภาพคลื่นไฟฟ้าหัวใจและทำให้การวินิจฉัยเข้าถึงได้จากทุกพื้นที่
              </Heading>
            </Box>
          </Stack>
        </Box>

        {/* origin block */}
        <Box
          bgColor="white"
          borderRadius="xl"
          py={{ base: 8, md: 10 }}
          px={{ base: 6, md: 12, lg: 16 }}
          mb={4}
        >
          <Stack
            direction={{ base: "column", md: "row" }}
            gap={{ base: 2, md: 12, lg: 16 }}
          >
            <Box minW={48}>
              <Heading
                as="h3"
                fontSize={{ base: "xl", md: "2xl" }}
                color="secondary.400"
                lineHeight="9"
              >
                จุดเริ่มต้นของ
                <Text as="span" display={{ base: "inline-block", md: "block" }}>
                  หทัย AI
                </Text>
              </Heading>
            </Box>

            <Box>
              <Stack
                direction="column"
                gap={{ base: 2, md: 4 }}
                fontWeight={400}
                color="gray.900"
              >
                <Text as="p">
                  การใช้คลื่นไฟฟ้าหัวใจ (Electrocardiogram, ECG)
                  เป็นหนึ่งในวิธีพื้นฐานที่แพทย์ใช้ในการตรวจสอบความผิดปกติของหัวใจในปัจจุบัน
                  ด้วยราคาและความรวดเร็วในการวัดผลทำให้แพทย์สามารถนำไปใช้ในการตรวจสอบในลำดับถัดไปได้อย่างมีประสิทธิภาพมากยิ่งขึ้น
                  แต่ทั้งนี้การเข้าถึงแพทย์ที่จะเข้าใจภาพคลื่นไฟฟ้าหัวใจเป็นไปได้ยาก
                  การใช้โมเดลทำนายที่มีประสิทธิภาพจึงมีโอกาสเข้ามามีส่วนในการช่วยแพทย์ในการอ่านภาพคลื่นไฟฟ้าหัวใจได้อย่างมีประสิทธิภาพมากยิ่งขึ้น
                </Text>
                <Text as="p">
                  ในปัจจุบันหทัย AI
                  สามารถอ่านภาพและประเมินความเสี่ยงของการมีแผลเป็นในกล้ามเนื้อหัวใจ
                  (Myocardial Scar)
                  และประเมินเมื่อประสิทธิภาพหัวใจห้องล่างซ้ายต่ำกว่าปกติ (LVEF
                  &lt; 40, 50) จากภาพคลื่นไฟฟ้าหัวใจแบบ 12 ลีด
                  และนอกจากนั้นยังมีแบบประเมินความเสี่ยงของการเกิดโรคหัวใจแบบต่าง
                  ๆ ได้แก่ การมีแผลเป็นในกล้ามเนื้อหัวใจ (Myocardial Scar)
                  หลอดเลือดหัวใจตีบหรือตัน (CAD)
                  และประสิทธิภาพหัวใจห้องล่างซ้ายต่ำกว่าปกติ (LVEF &lt; 40, 50)
                  ที่คำนวณมาจากสถิติของประชากรอีกด้วย
                </Text>
                <Text as="p">
                  คณะวิจัยจึงเล็งเห็นความสำคัญของการนำโมเดลมาพัฒนาต่อยอดเป็นแอปพลิเคชันและพัฒนาแอปพลิเคชันหทัย
                  AI ขึ้นมา
                  เพื่อให้แพทย์หรือผู้ป่วยสามารถเข้าถึงการวินิจฉัยขั้นพื้นฐานได้จากทุกพื้นที่ในประเทศไทย
                  นอกจากนั้นแอปพลิเคชันยังสามารถเพิ่มความตระหนักรู้ถึงความเสี่ยงในการเป็นโรคในกลุ่มประชากร
                  เพื่อได้รับการประเมินความเสี่ยงและได้รับการวินิจฉัยที่ทันท่วงทีก่อนเป็นโรคหัวใจร้ายแรง
                </Text>
              </Stack>
            </Box>
          </Stack>
        </Box>

        {/* team */}
        <Box
          bgColor="white"
          borderRadius="xl"
          py={{ base: 8, md: 10 }}
          px={{ base: 6, md: 12, lg: 16 }}
          mb={12}
        >
          <Stack
            direction={{ base: "column", md: "row" }}
            gap={{ base: 2, md: 12, lg: 16 }}
          >
            <Box minW={48}>
              <Heading
                as="h3"
                fontSize={{ base: "xl", md: "2xl" }}
                color="secondary.400"
                lineHeight="9"
              >
                คณะทำงาน
              </Heading>
            </Box>

            <Box>
              <Stack
                direction="column"
                gap={{ base: 2, md: 4 }}
                fontWeight={400}
                color="gray.900"
              >
                <Text as="p">
                  หทัย AI
                  เป็นแอปพลิเคชันที่สร้างขึ้นจากความร่วมมือระหว่างห้องทดลองของ{" "}
                  <Text as="b">ศ.นพ.รุ่งโรจน์ กฤตยพงษ์</Text> ศูนย์โรคหัวใจ และ{" "}
                  <Text as="b">ดร.ฐิติพัทธ อัชชะกุลวิสุทธิ์​</Text>{" "}
                  ภาควิชาวิศวกรรมชีวการแพทย์ มหาวิทยาลัยมหิดล
                  เพื่อช่วยให้การวินิจฉัยโรคหัวใจจากภาพคลื่นไฟฟ้าหัวใจทำได้อย่างมีประสิทธิภาพและเข้าถึงได้จากทุกพื้นที่
                  เพื่อใช้ในการรักษาโรคหัวใจกับผู้ป่วยในอนาคต
                  แอปพลิเคชันนี้จัดทำขึ้นเพื่อใช้ในการวิจัยและเพื่อให้แพทย์และประชาชนเข้าถึงโมเดลการทำนายรูปแบบต่าง
                  ๆ
                </Text>
                <Text as="p">
                  ในปัจจุบันแอปพลิเคชันไม่มีการเก็บข้อมูลของการอัปโหลดไฟล์ใด ๆ
                </Text>
              </Stack>
            </Box>
          </Stack>
        </Box>

        {/* logos */}
        <Stack
          direction={{ base: "column", md: "row" }}
          gap={{ base: 4, md: 0 }}
        >
          <Box>
            <Image
              src="/images/si-logo.png"
              alt="SI Logo"
              w={{ base: "90%", md: "80%", lg: "65%" }}
              mx="auto"
            />
          </Box>
          <Box>
            <Image
              src="/images/mu-logo.png"
              alt="MU Logo"
              w={{ base: "90%", md: "80%", lg: "65%" }}
              mx="auto"
            />
          </Box>
        </Stack>
      </Box>

      {/* contact containeer */}
      <Box position="relative">
        {/* another hacky bg */}
        <Box
          bg="gray.100"
          w="100vw"
          h="100%"
          position="absolute"
          top={0}
          left={0}
          zIndex={-1}
          marginLeft="calc((100% - 100vw) / 2)"
        />

        {/* contact itself */}
        <Box
          borderRadius="xl"
          py={{ base: 8, md: 10 }}
          px={{ base: 6, md: 12, lg: 16 }}
        >
          <Stack
            direction={{ base: "column", md: "row" }}
            gap={{ base: 2, md: 12, lg: 16 }}
          >
            <Box minW={48}>
              <Heading
                as="h3"
                fontSize={{ base: "xl", md: "2xl" }}
                color="secondary.400"
                lineHeight="9"
              >
                ติดต่อ
              </Heading>
            </Box>

            <Box>
              <Stack direction="column" fontWeight={400} color="gray.900">
                <Heading as="h6" size="sm" color="secondary.400">
                  อาคารศูนย์โรคหัวใจสมเด็จพระบรมราชินีนาถ โรงพยาบาลศิริราช ชั้น
                  9
                </Heading>
                <Text as="p" color="secondary.400">
                  แผนที่การเดินทาง{" "}
                  <Link
                    href="https://www.siphhospital.com/th/contact/contact"
                    isExternal
                  >
                    https://www.siphhospital.com/th/contact/contact
                  </Link>
                </Text>
              </Stack>
            </Box>
          </Stack>
        </Box>
      </Box>
    </Layout>
  );
};

export default About;
