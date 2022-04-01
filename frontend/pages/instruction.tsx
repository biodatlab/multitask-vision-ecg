import {
  Heading,
  Text,
  Stack,
  OrderedList,
  ListItem,
  Box,
} from "@chakra-ui/react";
import Layout from "../components/layout";

const Manual = () => {
  return (
    <Layout>
      <Stack direction="column" gap={2} pt={6}>
        <Heading as="h1">วิธีการใช้งาน</Heading>
        <Heading as="h2" size="md">
          ทำนายผลคลื่นไฟฟ้าหัวใจ
        </Heading>
        <Box>
          <OrderedList pl={4}>
            <ListItem>
              อัพโหลดหรือลากไฟล์ที่ต้องการไปยังพื้นที่
              ผู้ใช้สามารถกดกากบาทที่มุมขวาบนเพื่อลบไฟล์เพื่อเลือกใหม่ได้
              <br />
              [ภาพ]
            </ListItem>
            <ListItem>กดปุ่มทำนาย และรอโมเดลทำนายผล</ListItem>
            <ListItem>
              อ่านผลการทำนาย
              <br />
              [ภาพ]
            </ListItem>
          </OrderedList>
        </Box>
        <Heading as="h2" size="md">
          แบบประเมินความเสี่ยงภาวะโรคหัวใจล้มเหลว
        </Heading>
        <Box>
          <OrderedList pl={4}>
            <ListItem>
              กรอกข้อมูลผ่านแบบฟอร์ม
              <br />
              [ภาพ]
            </ListItem>
            <ListItem>กดทำนายผลเพื่อวัดความเสี่ยง</ListItem>
            <ListItem>
              อ่านผลการทำนาย
              <br />
              [ภาพ]
            </ListItem>
          </OrderedList>
        </Box>
      </Stack>
    </Layout>
  );
};

export default Manual;
