import {
  Alert,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  Box,
  Container,
  Flex,
  Heading,
  Stack,
  Text,
} from "@chakra-ui/react";
import type { NextPage } from "next";
import dynamic from "next/dynamic";
import { useRef, useState } from "react";
import Dropzone from "../components/ecg/dropzone";
import ModelDescription from "../components/ecg/modelDescription";
import Layout from "../components/layout";
import Prediction from "../components/shared/prediction";

// -- DYNAMIC IMPORT
const DownloadResultButton = dynamic(
  () => import("../components/shared/downloadResultButton"),
  { ssr: false }
);

// -- TYPES
export interface prediction {
  title: string;
  description?: string;
  risk_level: string;
  probability: number;
  average?: number;
}

export type predictionResult = Array<prediction> | { error: string } | null;

// -- MAIN
const Ecg: NextPage = () => {
  const [result, setResult] = useState(null as predictionResult);
  const [isLoadingResult, setIsLoadingResult] = useState(false);

  const predictionContainer = useRef<HTMLDivElement>(null);
  const resultImageContainer = useRef<HTMLDivElement>(null);

  return (
    <Layout>
      <Box my={12}>
        {/* main content and dropzone */}
        <Box position="relative" textAlign="center" borderRadius="3xl" py={10}>
          <Box
            position="absolute"
            borderRadius="3xl"
            top={0}
            left={0}
            w="100%"
            h={{ base: "40em", md: "29.5em" }}
            backgroundColor="secondary.50"
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
              AI ประเมินความเสี่ยง
              <br />
              จากภาพสแกนคลื่นไฟฟ้าหัวใจ
            </Heading>

            <Box maxW="container.sm" px={[0, 20]}>
              <Text mb={6}>
                AI
                ทำนายความน่าจะเป็นของการมีรอยแผลเป็นในหัวใจและค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายจากภาพสแกนคลื่นไฟฟ้าหัวใจ
                (Electrocardiogram, ECG) แบบ 12 Lead
              </Text>
              <Heading as="h6" fontSize="md" color="secondary.400" mb={1}>
                วิธีใช้งาน
              </Heading>
              <Text mb={6}>
                1. อัปโหลดภาพสแกน ECG ในบริเวณกรอบสี่เหลี่ยม
                <br />
                2. กดปุ่ม
                <Text as="span" color="primary.300" fontWeight={600}>
                  ทำนายผล
                </Text>{" "}
                และรอโมเดลทำนายผล
              </Text>

              <Box left={[0, "50%"]} right={[0, "50%"]}>
                <Dropzone
                  onClearFile={() => setResult(null)}
                  onSubmit={(f) => {
                    setIsLoadingResult(true);
                    console.log(f);

                    const formdata = new FormData();
                    formdata.append("file", f, f.name);

                    fetch("/api/predict", {
                      method: "POST",
                      body: formdata,
                    })
                      .then((res) => {
                        return res.json();
                      })
                      .then((resJson) => {
                        setResult(resJson);
                        console.log("resJson", resJson);
                      })
                      .finally(() => {
                        setIsLoadingResult(false);

                        predictionContainer.current?.scrollIntoView({
                          behavior: "smooth",
                          block: "start",
                        });
                      });
                  }}
                  isLoading={isLoadingResult}
                />
              </Box>
            </Box>
          </Container>
        </Box>

        {/* if prediction result valid */}
        {result && Array.isArray(result) && (
          <Box ref={predictionContainer} mb={-12}>
            <Box ref={resultImageContainer} mb={16}>
              <Stack direction="column" gap={4} pt={10}>
                <Flex justify="space-between" align="center" mb={6}>
                  <Heading as="h4" fontSize="2xl" color="secondary.400">
                    ผลทำนาย
                  </Heading>
                  <DownloadResultButton
                    targetRef={resultImageContainer.current}
                  />
                </Flex>
                <Prediction predictionResult={result} />
              </Stack>

              <Box pt={6} textAlign="center">
                <Text fontSize="xs">
                  <Text as="span" fontWeight="semibold">
                    หมายเหตุ:&nbsp;
                  </Text>
                  ผลการทำนายเป็นการประมาณจากโมเดลเพื่อใช้เป็นการประเมินเบื้องต้นสำหรับการรักษาเท่านั้น
                  ไม่สามารถใช้ผลลัพธ์แทนแพทย์ผู้เชี่ยวชาญได้
                </Text>
              </Box>
            </Box>

            {/* model description */}
            <Box>
              <Flex
                marginLeft="calc(50% - 50vw)"
                width="100vw"
                backgroundColor="gray.100"
              >
                <Container maxW="container.lg">
                  <ModelDescription />
                </Container>
              </Flex>
            </Box>
          </Box>
        )}

        {/* if prediction result error */}
        {result && !Array.isArray(result) && (
          <Box>
            <Alert
              status="error"
              variant="subtle"
              colorScheme="gray"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              height="200px"
              textAlign="center"
              borderRadius="3xl"
            >
              <AlertIcon boxSize="40px" mr={0} />
              <AlertTitle mt={4} mb={1} fontSize="lg">
                ไม่สามารถทำนายได้
              </AlertTitle>
              <AlertDescription maxWidth="sm">
                อาจมีข้อผิดพลาดเกี่ยวกับรูปภาพหรือไฟล์ที่ส่งมา
                กรุณาลองใหม่อีกครั้งหรือติดต่อเรา
              </AlertDescription>
            </Alert>
          </Box>
        )}
      </Box>
    </Layout>
  );
};

export default Ecg;
