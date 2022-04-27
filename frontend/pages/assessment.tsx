import {
  Box,
  Code,
  Container,
  Grid,
  GridItem,
  Heading,
  Stack,
  Text,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import Form from "../components/assessment/form";
import { PredictionCard } from "../components/ecg/prediction";
import Layout from "../components/layout";
import { prediction } from "./ecg";

const CodeProps = {
  w: "100%",
  backgroundColor: "white",
  borderRadius: "xl",
  p: 6,
};

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

      {/* results */}
      {results.length > 0 && (
        <>
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
                  <Grid
                    templateColumns={{
                      base: "repeat(1, 1fr)",
                      lg: "repeat(2, 1fr)",
                    }}
                    gap={6}
                  >
                    {results.map((data) => (
                      <GridItem key={data.title}>
                        <PredictionCard data={data} />
                      </GridItem>
                    ))}
                  </Grid>
                </Box>
              </Stack>
            </Container>
          </Box>
          {/* model description */}
          <Box>
            <Container maxW="container.lg">
              <VStack gap={6} alignItems="flex-start" py={12} pb={16}>
                <Heading
                  as="h5"
                  fontWeight="semibold"
                  size="md"
                  color="secondary.400"
                >
                  สูตรการคำนวณความเสี่ยง
                </Heading>
                <VStack gap={2} fontSize="sm" px={6} alignItems="flex-start">
                  <Text>
                    สูตรการคำนวณความเสี่ยงนี้ถูกสร้างขึ้นมาจากข้อมูลทางคลินิกจากกลุ่มประชากรที่เข้ามาตรวจ
                    ณ​ ศูนย์โรคหัวใจ โรงพยาบาลศิริราช
                    โดยใช้ฟังก์ชันการเรียนรู้แบบถอดถอดแบบโลจิสติกส์ (Logistic
                    regression)
                    เพื่อทำนายความน่าจะเป็นของการมีโรคหัวใจประเภทต่างๆ
                    โดยสมการที่ใช้คำนวณเป็นไปตามสูตรต่อไปนี้
                  </Text>

                  <Heading as="h6" fontWeight="semibold" fontSize="md">
                    ความน่าจะเป็นของการมีแผลเป็นในกล้ามเนื้อหัวใจ (Myocardial
                    Infarction, MI)
                  </Heading>
                  <Code {...CodeProps}>
                    logit_scar = (-1.253690 + (-1.289421 * female) + (1.267990*
                    history_mi) + (0.852316 * history_ptca_cabg) + (1.188963 *
                    history_cad) + (0.875971 * history_hf) + (0.394268 *
                    history_ckd))
                  </Code>

                  <Heading as="h6" fontWeight="semibold" fontSize="md">
                    ความน่าจะเป็นของการเป็นโรคหลอดเลือดหัวใจตีบหรือตัน (Coronary
                    Artery Disease, CAD)
                  </Heading>
                  <Code {...CodeProps}>
                    logit_cad_scar = (-1.838088 + (-1.257548 * female) +
                    (0.290813 * diabetes_mellitus) + (1.428430 * history_mi).
                    +(0.915726 * history_ptca_cabg) + (1.477169 * history_cad) +
                    (0.519559 * history_hf) + (0.359040 * history_ckd))
                  </Code>

                  <Heading as="h6" fontWeight="semibold" fontSize="md">
                    ความน่าจะเป็นที่ประสิทธิภาพการบีบตัวหัวใจห้องล่างซ้ายตำ่กว่า
                    40% (LVEF &lt; 40)
                  </Heading>
                  <Code {...CodeProps}>
                    logit_lvef40 = (-1.986239 + (-0.573404 * age_above_65) +
                    (-0.537664 * female) + (-0.771137 * bmi_above_25) +
                    (-0.533213 * hypertension) + (0.714925 * history_cad) +
                    (2.065291 * history_hf) + (0.619290 * history_ckd))
                  </Code>

                  <Heading as="h6" fontWeight="semibold" fontSize="md">
                    ความน่าจะเป็นที่ประสิทธิภาพการบีบตัวหัวใจห้องล่างซ้ายตำ่กว่า
                    50% (LVEF &lt; 50)
                  </Heading>
                  <Code {...CodeProps}>
                    logit_lvef50 = (-1.468317 + (-0.438355 * age_above_65) +
                    (-0.797398 * female) + (-0.3547618 * bmi_above_25) +
                    (-0.350557 * hypertension) + (0.534031 * history_mi) +
                    (0.787748 * history_cad) + (1.831592 * history_hf))
                  </Code>

                  <Heading as="h6" fontWeight="semibold" fontSize="md">
                    เมื่อได้ Logit มาแล้ว นำไปคำนวณหาความน่าจะเป็นจากสูตร
                  </Heading>
                  <Code {...CodeProps}>
                    probability = 1 / (1 + exp(-logit))
                  </Code>
                </VStack>
              </VStack>
            </Container>
          </Box>
        </>
      )}

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
