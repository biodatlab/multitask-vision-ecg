import {
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

const ModelDescription = () => {
  return (
    <VStack gap={6} alignItems="flex-start" py={12} pb={16}>
      <Heading as="h5" fontWeight="semibold" size="md" color="secondary.400">
        วิธีการสร้างและวัดผลความแม่นยำของโมเดล
      </Heading>
      <VStack gap={2} fontSize="sm" px={6}>
        <Text>
          โมเดลทำนายผลของภาพสแกนคลื่นไฟฟ้าหัวใจ (Electrocardiogram, ECG) แบบ 12
          Lead ที่ใช้ทำนายนี้ถูกสร้างขึ้นมาจากฐานข้อมูลภาพสแกนคลื่นไฟฟ้า
          หัวใจจำนวน 2,100 ภาพ ที่เก็บจากศูนย์โรคหัวใจโรงพยาบาลศิริราช
          โดยเป็นข้อมูลของคลื่นไฟฟ้าหัวใจที่ไม่มีแผลเป็น 1,100 ภาพ และมีแผลเป็น
          1,000 ภาพ
        </Text>

        <Text>
          ในการสร้างโมเดลการทำนาย ข้อมูลถูกแบ่งเป็นชุดข้อมูลสำหรับสร้างโมเดล,
          วัดผลระหว่างคำนวณโมเดล และทดสอบโมเดล ด้วยสัดส่วน 80:10:10
          ทั้งนี้โมเดลอาจมีความผิดพลาดในการทำนายผลได้
          ผู้ใช้งานสามารถดูความถูกต้องของการทดสอบโมเดลเบื้องต้นได้ตามตารางด้านล่าง
          (ผลเป็น %)
        </Text>
      </VStack>

      <Heading as="h5" fontWeight="semibold" size="md" color="secondary.400">
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
  );
};

export default ModelDescription;
