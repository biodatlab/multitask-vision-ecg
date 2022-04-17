import { Container, Flex, Spacer } from "@chakra-ui/react";
import Footer from "./footer";
// import { useAuthState } from "react-firebase-hooks/auth";
// import LoginModal from "../components/loginModal";
import Navbar from "./navbar";
// import { auth, logOut } from "../utils/firebase/clientApp";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  // const {
  //   isOpen: isOpenLoginModal,
  //   onClose: onCloseLoginModal,
  //   onOpen: onOpenLoginModal,
  // } = useDisclosure();

  // const [user, loading, error] = useAuthState(auth!);

  // useEffect(() => {
  //   console.log('user', user);
  // }, [user]);

  return (
    <Flex
      h="100vh"
      flexDirection="column"
      background="#fefefd"
      color="gray.600"
      // using 100vw inside this flex will cause the page to
      // have scroll-x, this will prevent that
      overflowX="hidden"
    >
      {/* <LoginModal isOpen={isOpenLoginModal} onClose={onCloseLoginModal} /> */}
      <Navbar
        // onClickLogin={onOpenLoginModal}
        // onClickLogout={logOut}
        // isLoadingUserStatus={loading}
        // isLoggedIn={!!user}
        onClickLogin={() => {}}
        onClickLogout={() => {}}
        isLoadingUserStatus={false}
        isLoggedIn={false}
      />
      <Container maxW="container.lg">{children}</Container>
      <Spacer />
      <Footer />
    </Flex>
  );
};

export default Layout;
