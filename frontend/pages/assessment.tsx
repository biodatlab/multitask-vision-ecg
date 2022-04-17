import { Box, Container, Heading, Text } from "@chakra-ui/react";
import Form from "../components/assessment/form";
import Layout from "../components/layout";

const Assessment = () => {
  return (
    <Layout>
      <Box my={12}>
        {/* main content */}
        <Box
          position="relative"
          textAlign="center"
          borderRadius="3xl"
          pt={10}
          pb={12}
          mb={8}
        >
          <Box
            position="absolute"
            borderRadius="3xl"
            top={0}
            left={0}
            w="100%"
            h="100%"
            backgroundColor="white"
          />
          <Container maxW="container.sm" position="relative">
            <Heading
              as="h1"
              fontSize={40}
              lineHeight="tall"
              fontWeight="semibold"
              color="secondary.400"
              mb={2}
            >
              แบบประเมินความเสี่ยง
              <br />
              ภาวะโรคหัวใจต่าง ๆ
            </Heading>

            <Box maxW="container.sm" px={[0, 25]}>
              <Text>
                แบบประเมินความเสี่ยงของการมีรอยแผลเป็นในหัวใจ (Myocardial Scar)
                โรคหลอดเลือดแดงของหัวใจตีบหรือตัน (Coronary Artery Disease, CAD)
                และการบีบตัวของหัวใจห้องล่างซ้ายผิดปกติ (LVEF)
                สำหรับประชาชนทั่วไปหรือผู้ที่มีความเสี่ยง
              </Text>
            </Box>
          </Container>
        </Box>

        {/* questions form */}
        <Box
          maxW="2xl"
          backgroundColor="white"
          borderRadius="2xl"
          boxShadow="lg"
          py={10}
          px={14}
          mx="auto"
        >
          <Form onCalculate={(results) => console.log(results)} />
        </Box>
      </Box>

      {/* hacky back faint secondary bg */}
      <Box
        zIndex={-1}
        position={"fixed"}
        top={0}
        left={0}
        width="100vw"
        height="100vh"
        background="secondary.50"
      />
    </Layout>
  );
};

export default Assessment;
