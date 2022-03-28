import { Heading, Text, Stack, OrderedList, ListItem } from "@chakra-ui/react";
import Layout from "../components/layout";

const Manual = () => {
  return (
    <Layout>
      <Stack color="gray.600" direction="column" gap={2} pt={6}>
        <Heading as="h1">วิธีการใช้งาน</Heading>
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
      </Stack>
    </Layout>
  );
};

export default Manual;
