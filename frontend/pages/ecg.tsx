import {
  Alert,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  Box,
  Container,
  Flex,
  Heading,
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  VStack,
} from "@chakra-ui/react";
import type { NextPage } from "next";
import { useState } from "react";
import Dropzone from "../components/dropzone";
import Layout from "../components/layout";
import Prediction from "../components/prediction";

export interface prediction {
  title: string;
  description: string;
  risk_level: string;
  probability: number;
  average: number;
}

export type predictionResult = Array<prediction> | { error: string } | null;

const Ecg: NextPage = () => {
  const [result, setResult] = useState(null as predictionResult);
  const [isLoadingResult, setIsLoadingResult] = useState(false);

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
            h="29.5em"
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
              <Text mb={8}>
                AI
                ทำนายความน่าจะเป็นของการมีรอยแผลเป็นในหัวใจและค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายจากภาพสแกนคลื่นไฟฟ้าหัวใจ
                (Electrocardiogram, ECG) แบบ 12 Lead
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
          <Box mb={-12}>
            <Box mb={16}>
              <Prediction predictionResult={result} />
              <Box maxW="md" pt={6} mx="auto">
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
                  <VStack gap={6} alignItems="flex-start" py={12} pb={16}>
                    <Heading
                      as="h5"
                      fontWeight="semibold"
                      size="md"
                      color="secondary.400"
                    >
                      วิธีการสร้างและวัดผลความแม่นยำของโมเดล
                    </Heading>
                    <VStack gap={2} fontSize="sm" px={6}>
                      <Text>
                        โมเดลทำนายผลของภาพสแกนคลื่นไฟฟ้าหัวใจ
                        (Electrocardiogram, ECG) แบบ 12 Lead
                        ที่ใช้ทำนายนี้ถูกสร้างขึ้นมาจากฐานข้อมูลภาพสแกนคลื่นไฟฟ้า
                        หัวใจจำนวน 2,100 ภาพ
                        ที่เก็บจากศูนย์โรคหัวใจโรงพยาบาลศิริราช
                        โดยเป็นข้อมูลของคลื่นไฟฟ้าหัวใจที่ไม่มีแผลเป็น 1,100 ภาพ
                        และมีแผลเป็น 1,000 ภาพ
                      </Text>

                      <Text>
                        ในการสร้างโมเดลการทำนาย
                        ข้อมูลถูกแบ่งเป้นชุดข้อมูลสำหรับสร้างโมเดล,
                        วัดผลระหว่างคำนวณโมเดล และทดสอบโมเดล ด้วยสัดส่วน
                        80:10:10 ทั้งนี้โมเดลอาจมีความผิดพลาดในการทำนายผลได้
                        ผู้ใช้งานสามารถดูความถูกต้องของการทดสอบโมเดลเบื้องต้นได้ตามตารางด้านล่าง
                        (ผลเป็น %)
                      </Text>
                    </VStack>

                    <Heading
                      as="h5"
                      fontWeight="semibold"
                      size="md"
                      color="secondary.400"
                    >
                      การวัดผลความแม่นยำของโมเดล
                    </Heading>
                    <TableContainer
                      backgroundColor="white"
                      borderRadius="2xl"
                      p={2}
                      pt={4}
                      sx={{
                        marginX: "auto !important",
                      }}
                    >
                      <Table variant="simple">
                        <Thead>
                          <Tr>
                            <Th>โมเดล</Th>
                            <Th isNumeric>
                              ความแม่นยำ
                              <br />
                              (Accuracy)
                            </Th>
                            <Th isNumeric>
                              ความจำเพาะ
                              <br />
                              (Specificity)
                            </Th>
                            <Th isNumeric>
                              ความไว
                              <br />
                              (Sensitivity)
                            </Th>
                            <Th isNumeric>AUROC</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          <Tr>
                            <Td>แผลเป็นในหัวใจ (Mycardial scar)</Td>
                            <Td isNumeric>80.5</Td>
                            <Td isNumeric>85.0</Td>
                            <Td isNumeric>80.5</Td>
                            <Td isNumeric>89.1</Td>
                          </Tr>
                          <Tr>
                            <Td>LVEF &lt; 40</Td>
                            <Td isNumeric>89.0</Td>
                            <Td isNumeric>88.6</Td>
                            <Td isNumeric>89.0</Td>
                            <Td isNumeric>92.9</Td>
                          </Tr>
                          <Tr>
                            <Td borderBottom="none">LVEF &lt; 50</Td>
                            <Td borderBottom="none" isNumeric>
                              84.8
                            </Td>
                            <Td borderBottom="none" isNumeric>
                              84.2
                            </Td>
                            <Td borderBottom="none" isNumeric>
                              84.8
                            </Td>
                            <Td borderBottom="none" isNumeric>
                              90.5
                            </Td>
                          </Tr>
                        </Tbody>
                      </Table>
                    </TableContainer>
                  </VStack>
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
