import { Box, Container, Heading, Stack, Text, VStack } from "@chakra-ui/react";
import { useState } from "react";
import Form from "../components/assessment/form";
import { PredictionCard } from "../components/ecg/prediction";
import Layout from "../components/layout";
import { prediction } from "./ecg";

const Assessment = () => {
  const [results, setResults] = useState<Array<prediction>>([]);

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
          <Form onCalculate={(r) => setResults(r)} />
        </Box>
      </Box>

      {results.length > 0 && (
        <Box position="relative" textAlign="center" py={10}>
          <Box
            position="absolute"
            top={0}
            left={0}
            marginLeft="calc(50% - 50vw)"
            w="100vw"
            h="100%"
            backgroundColor="white"
          />
          <Container maxW="container.lg" position="relative">
            <Stack direction="column" gap={4} mt={10} alignItems="flex-start">
              <Heading as="h4" fontSize="2xl" color="secondary.400" mb={6}>
                ผลการคำนวณ
              </Heading>
              <Box w="100%">
                <VStack gap={8}>
                  {results.map((data) => (
                    <PredictionCard key={data.title} data={data} />
                  ))}
                </VStack>
              </Box>
            </Stack>
          </Container>
        </Box>
      )}

      {/* hacky back faint secondary bg */}
      <Box
        zIndex={-1}
        position={"fixed"}
        top={0}
        left={0}
        width="100vw"
        height="100vh"
        background="secondary.100"
      />
    </Layout>
  );
};

export default Assessment;
